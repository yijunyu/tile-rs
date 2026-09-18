// Vulkan compute harness for tile-rs SPIR-V output.
//
// Satisfies the same contract as assets/harness/metal.swift, so `run.rs` reads one output
// format whichever device it drove:
//
//   argv: <kernel.spv> <entry> <dtype> <iters> <grid> <threads> <in-counts,csv>
//         <out-count> <scalars,csv> [sweep-widths,csv]
//   out:  #device NAME / #threadgroup N / #warmup N / #us T... / values / #c control...
//         or, in sweep mode, #sweep WIDTH MEDIAN_US per width
//
// Build (the caller does this; documented here so it can be reproduced by hand):
//   glslangValidator -V --target-env vulkan1.2 -S comp kernel.comp -o kernel.spv
//   clang -O2 -I<vulkan-include> -L<vulkan-lib> -lvulkan -o vkrun vulkan.c
//
// The input generator matches metal.swift's and run.rs::input_values_for exactly --
// (i + 7*b + seed) % 17 * 0.25 - 2.0 + i * 1e-4 -- because a reference fed different data
// from the kernel compares two unrelated numbers.
//
// It found #016 the first time it ran: the softmax returned 224 non-finite values of 1024.
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(x) do{ VkResult r_=(x); if(r_){ fprintf(stderr,"vulkan: %s:%d rc=%d\n",__FILE__,__LINE__,r_); exit(3);} }while(0)
#define MAXBUF 8

static uint32_t *load_spv(const char *p, size_t *n){
  FILE *f=fopen(p,"rb"); if(!f){ fprintf(stderr,"vulkan: cannot open %s\n",p); exit(3);}
  fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
  uint32_t *b=malloc(s);
  if(fread(b,1,s,f)!=(size_t)s){ fprintf(stderr,"vulkan: short read %s\n",p); exit(3);}
  fclose(f); *n=(size_t)s; return b;
}
static int csv(const char*s,uint32_t*out,int max){
  int n=0; if(!s||!*s) return 0;
  const char*p=s; while(*p&&n<max){ out[n++]=(uint32_t)strtoul(p,0,10);
    const char*c=strchr(p,','); if(!c)break; p=c+1;} return n;
}
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec*1e6 + t.tv_nsec/1e3; }
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b;
  return x<y?-1:(x>y?1:0); }

// Same rule as metal.swift and run.rs::input_values_for.
// The CONTROL arm (seed != 0) scales and offsets as well as reseeding, for the reason
// metal.swift states at length: a reseed only PERMUTES the row, so a max or absmax over 17
// or more elements returns nearly the same number, and at f16 precision the difference
// vanished entirely -- the control arm then reported that a correct kernel was ignoring
// its inputs.
static float input_value(int b,int i,int seed){
  float k = seed ? 1.37f : 1.0f, o = seed ? 0.11f : 0.0f;
  return (float)(((i + 7*b + seed) % 17) * 0.25 - 2.0) * k + o + (float)i * 1e-4f;
}

// File-scope state so the two helpers below can be plain C functions. They were nested,
// which is a GNU extension clang does not take by default.
static VkDevice g_dev; static VkQueue g_q; static VkCommandBuffer g_cb;
static VkPipeline g_pipe; static VkPipelineLayout g_pl; static VkDescriptorSet g_ds;
static uint32_t g_pcsize, g_pcvals[MAXBUF], g_grid, g_inc[MAXBUF], g_outc;
static int g_nin;
static void *g_map[MAXBUF];

static double dispatch_once(void){
  CK(vkResetCommandBuffer(g_cb,0));
  VkCommandBufferBeginInfo cbbi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  CK(vkBeginCommandBuffer(g_cb,&cbbi));
  vkCmdBindPipeline(g_cb,VK_PIPELINE_BIND_POINT_COMPUTE,g_pipe);
  vkCmdBindDescriptorSets(g_cb,VK_PIPELINE_BIND_POINT_COMPUTE,g_pl,0,1,&g_ds,0,0);
  vkCmdPushConstants(g_cb,g_pl,VK_SHADER_STAGE_COMPUTE_BIT,0,g_pcsize,g_pcvals);
  vkCmdDispatch(g_cb,g_grid,1,1);
  CK(vkEndCommandBuffer(g_cb));
  VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,
    .pCommandBuffers=&g_cb};
  double t0=now_us();
  CK(vkQueueSubmit(g_q,1,&si,VK_NULL_HANDLE));
  CK(vkQueueWaitIdle(g_q));
  return now_us()-t0;
}

// Half support. The SPIR-V backend emits `float16_t` buffers for f16 kernels, and this
// harness filled every buffer with f32 -- so the whole f16 sweep was refused rather than
// measured. `_Float16` is native on arm64 clang, which is what builds this file.
//
// The rounding must match `run::f16_round` and metal.swift's `Float16(v)` exactly, or the
// three sides compute over different numbers, which is backlog #010 all over again.
static int g_half = 0;
static size_t elem_size(void){ return g_half ? 2 : 4; }

static void store_elem(void*base,uint32_t i,float v){
  if(g_half) ((_Float16*)base)[i] = (_Float16)v;
  else       ((float*)base)[i]    = v;
}
static float load_elem(const void*base,uint32_t i){
  return g_half ? (float)((const _Float16*)base)[i] : ((const float*)base)[i];
}

static void fill_inputs(int seed){
  for(int b=0;b<g_nin;b++){
    for(uint32_t i=0;i<g_inc[b];i++) store_elem(g_map[b],i,input_value(b,(int)i,seed));
  }
  memset(g_map[g_nin],0,(size_t)g_outc*elem_size());
}

int main(int argc,char**argv){
  if(argc<10){ fprintf(stderr,"usage: vkrun kernel.spv entry dtype iters grid threads "
                              "in-counts out-count scalars [sweep]\n"); return 2; }
  const char *spv=argv[1], *entry=argv[2], *dtype=argv[3];
  int iters=atoi(argv[4]); uint32_t grid=atoi(argv[5]); uint32_t threads=atoi(argv[6]);
  uint32_t inc[MAXBUF]; int nin=csv(argv[7],inc,MAXBUF);
  uint32_t outc=(uint32_t)atoi(argv[8]);
  uint32_t scal[MAXBUF]; int nscal=csv(argv[9],scal,MAXBUF);
  uint32_t sweep[16]; int nsweep = argc>10 ? csv(argv[10],sweep,16) : 0;
  (void)entry;   // entry is always "main" in GLSL
  g_half = (strcmp(dtype,"half")==0);

  VkApplicationInfo ai={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_2};
  VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&ai};
  VkInstance inst; CK(vkCreateInstance(&ici,0,&inst));
  uint32_t pdc=0; CK(vkEnumeratePhysicalDevices(inst,&pdc,0));
  if(!pdc){ fprintf(stderr,"vulkan: no physical device\n"); return 4; }
  VkPhysicalDevice *pds=malloc(pdc*sizeof(*pds)); CK(vkEnumeratePhysicalDevices(inst,&pdc,pds));
  VkPhysicalDevice pd=pds[0];
  VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd,&props);

  uint32_t qfc=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&qfc,0);
  VkQueueFamilyProperties*qf=malloc(qfc*sizeof(*qf));
  vkGetPhysicalDeviceQueueFamilyProperties(pd,&qfc,qf);
  uint32_t qi=UINT32_MAX;
  for(uint32_t i=0;i<qfc;i++) if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qi=i;break;}
  if(qi==UINT32_MAX){ fprintf(stderr,"vulkan: no compute queue\n"); return 4; }
  float prio=1.0f;
  VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex=qi,.queueCount=1,.pQueuePriorities=&prio};
  VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
  VkDevice dev; CK(vkCreateDevice(pd,&dci,0,&dev));
  VkQueue q; vkGetDeviceQueue(dev,qi,0,&q);

  int nbuf=nin+1;
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
  VkBuffer buf[MAXBUF]; VkDeviceMemory mem[MAXBUF]; void*map[MAXBUF];
  for(int b=0;b<nbuf;b++){
    size_t cnt = (b<nin) ? inc[b] : outc;
    VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size=cnt*elem_size(),.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    CK(vkCreateBuffer(dev,&bci,0,&buf[b]));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf[b],&mr);
    uint32_t mt=UINT32_MAX;
    for(uint32_t i=0;i<mp.memoryTypeCount;i++)
      if((mr.memoryTypeBits&(1u<<i))
         &&(mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
         &&(mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){mt=i;break;}
    if(mt==UINT32_MAX){ fprintf(stderr,"vulkan: no host-coherent memory\n"); return 4; }
    VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize=mr.size,.memoryTypeIndex=mt};
    CK(vkAllocateMemory(dev,&mai,0,&mem[b]));
    CK(vkBindBufferMemory(dev,buf[b],mem[b],0));
    CK(vkMapMemory(dev,mem[b],0,VK_WHOLE_SIZE,0,&map[b]));
  }

  size_t sz; uint32_t*code=load_spv(spv,&sz);
  VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize=sz,.pCode=code};
  VkShaderModule sm; CK(vkCreateShaderModule(dev,&smci,0,&sm));
  VkDescriptorSetLayoutBinding lb[MAXBUF];
  for(int b=0;b<nbuf;b++) lb[b]=(VkDescriptorSetLayoutBinding){.binding=(uint32_t)b,
    .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,
    .stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
  VkDescriptorSetLayoutCreateInfo dlci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount=(uint32_t)nbuf,.pBindings=lb};
  VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev,&dlci,0,&dsl));
  uint32_t pcsize = (uint32_t)((nscal? nscal:1)*sizeof(uint32_t));
  VkPushConstantRange pcr={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=pcsize};
  VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount=1,.pSetLayouts=&dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
  VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev,&plci,0,&pl));
  VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},.layout=pl};
  VkPipeline pipe; CK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,0,&pipe));

  VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=(uint32_t)nbuf};
  VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps};
  VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev,&dpci,0,&dp));
  VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&dsl};
  VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev,&dsai,&ds));
  VkDescriptorBufferInfo bi[MAXBUF]; VkWriteDescriptorSet w[MAXBUF];
  for(int b=0;b<nbuf;b++){
    bi[b]=(VkDescriptorBufferInfo){.buffer=buf[b],.range=VK_WHOLE_SIZE};
    w[b]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,
      .dstBinding=(uint32_t)b,.descriptorCount=1,
      .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[b]};
  }
  vkUpdateDescriptorSets(dev,(uint32_t)nbuf,w,0,0);

  VkCommandPoolCreateInfo cpi={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qi};
  VkCommandPool cp; CK(vkCreateCommandPool(dev,&cpi,0,&cp));
  VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool=cp,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
  VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev,&cbai,&cb));

  uint32_t pcvals[MAXBUF];
  for(int i=0;i<nscal;i++) pcvals[i]=scal[i];
  if(!nscal) pcvals[0]=inc[0];

  g_dev=dev; g_q=q; g_cb=cb; g_pipe=pipe; g_pl=pl; g_ds=ds;
  g_pcsize=pcsize; g_grid=grid; g_outc=outc; g_nin=nin;
  for(int i=0;i<MAXBUF;i++){ g_pcvals[i]=pcvals[i]; g_inc[i]=inc[i]; g_map[i]=map[i]; }
  (void)g_dev;

  int warmup = iters/10 > 3 ? iters/10 : 3;
  fill_inputs(0);

  if(nsweep){
    // Sweep mode: the threadgroup is compiled into a SPIR-V shader, so unlike Metal this
    // harness cannot vary it per dispatch. It reports the one width it has, once, rather
    // than inventing measurements for widths it did not run.
    (void)threads;
    for(int i=0;i<warmup;i++) dispatch_once();
    double *ts=malloc(iters*sizeof(double));
    for(int i=0;i<iters;i++) ts[i]=dispatch_once();
    qsort(ts,iters,sizeof(double),cmpd);
    printf("#sweep %u %.4f\n", threads, ts[iters/2]);
    return 0;
  }

  for(int i=0;i<warmup;i++) dispatch_once();
  double *ts=malloc(iters*sizeof(double));
  for(int i=0;i<iters;i++) ts[i]=dispatch_once();

  printf("#device %s\n", props.deviceName);
  printf("#threadgroup %u\n", threads);
  printf("#warmup %d\n", warmup);
  for(int i=0;i<iters;i++) printf("#us %.4f\n", ts[i]);

  // Read back through the element type, so an f16 result is widened once here rather
  // than reinterpreted as f32 bits.
  float *ov=malloc(outc*sizeof(float));
  for(uint32_t i=0;i<outc;i++) ov[i]=load_elem(map[nin],i);

  // The control arm, same rule as metal.swift: refill from a DIFFERENT seed and dispatch
  // again. The output must change. If it does not, this harness is not moving data and
  // every number above describes something other than this dispatch.
  fill_inputs(1);
  dispatch_once();
  for(uint32_t i=0;i<outc;i++) printf("%.6e\n", ov[i]);
  for(uint32_t i=0;i<outc;i++) printf("#c %.6e\n", load_elem(map[nin],i));
  return 0;
}
