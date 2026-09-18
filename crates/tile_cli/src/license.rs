//! The license gate, and an honest account of what it can and cannot do.
//!
//! ## What the gate actually is
//!
//! A license check on a plaintext file sitting on the user's disk is decoration — and so
//! is encrypting the file and shipping the key beside it. A licensed user can dump the
//! whole corpus through `-s` itself; offline expiry is unenforceable against a copy
//! already decrypted. Anyone who tells you otherwise is selling obfuscation.
//!
//! So the gate here is on **delivery, not on the file**: the token is what lets a client
//! obtain and open a sealed corpus snapshot, and anything that must genuinely stay
//! private belongs behind a service rather than in a shipped artifact. What remains
//! client-side is honest product segmentation, and this module says so in its own doc
//! rather than implying more.
//!
//! What it *does* buy, and buys properly:
//!
//! * The corpus is **unreadable without a token** — it is sealed with XChaCha20-Poly1305,
//!   so it is not one `strings` away, and a corrupted or tampered snapshot fails to open
//!   rather than opening partially.
//! * A token is **unforgeable**: Ed25519 over a canonical payload, verified against a key
//!   compiled into the binary. No phone-home, so it works on an air-gapped box.
//! * Expiry and the feature set are **stated**, so `tile license status` is a real answer
//!   rather than a yes/no.

use std::fmt;
use std::path::PathBuf;

/// What a token grants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct License {
    pub subject: String,
    /// RFC-3339 date. Compared lexically, which is correct for this format.
    pub expires: String,
    pub features: Vec<String>,
    /// The 32-byte key that opens a sealed corpus, hex-encoded.
    pub content_key: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum LicenseError {
    Absent,
    Malformed(String),
    /// The signature does not verify against the embedded public key.
    UnknownKey,
    Tampered,
    Expired {
        on: String,
    },
}

impl fmt::Display for LicenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LicenseError::Absent => write!(
                f,
                "no license key installed. The curated corpus stays closed; everything \
                 else, including your own recorded attempts, works unchanged."
            ),
            LicenseError::Malformed(e) => write!(f, "the license file is malformed: {e}"),
            LicenseError::UnknownKey => write!(
                f,
                "the license is signed by an unknown key and will not be trusted"
            ),
            LicenseError::Tampered => write!(f, "the license has been tampered with"),
            LicenseError::Expired { on } => write!(f, "the license expired on {on}"),
        }
    }
}

pub fn path() -> PathBuf {
    crate::provision::home().join("license")
}

/// The bytes that are signed. Canonical and order-fixed: a signature over a
/// re-serialization that reorders fields verifies nothing.
pub fn payload(l: &License) -> String {
    format!(
        "tile-license/1\nsubject={}\nexpires={}\nfeatures={}\nkey={}\n",
        l.subject,
        l.expires,
        l.features.join(","),
        l.content_key
    )
}

pub fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err("odd-length hex".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

pub fn hex_encode(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Parse a license file: the payload, then `signature=<hex>`.
pub fn parse(text: &str) -> Result<(License, Vec<u8>), LicenseError> {
    let mut subject = None;
    let mut expires = None;
    let mut features = Vec::new();
    let mut key = None;
    let mut sig = None;
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t == "tile-license/1" {
            continue;
        }
        let Some((k, v)) = t.split_once('=') else {
            return Err(LicenseError::Malformed(format!(
                "not a key=value line: {t:?}"
            )));
        };
        match k {
            "subject" => subject = Some(v.to_string()),
            "expires" => expires = Some(v.to_string()),
            "features" => {
                features = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect()
            }
            "key" => key = Some(v.to_string()),
            "signature" => sig = Some(v.to_string()),
            other => return Err(LicenseError::Malformed(format!("unknown field {other:?}"))),
        }
    }
    let l = License {
        subject: subject.ok_or(LicenseError::Malformed("no subject".into()))?,
        expires: expires.ok_or(LicenseError::Malformed("no expires".into()))?,
        features,
        content_key: key.ok_or(LicenseError::Malformed("no key".into()))?,
    };
    let sig = sig.ok_or(LicenseError::Malformed("no signature".into()))?;
    let sig = hex_decode(&sig).map_err(LicenseError::Malformed)?;
    Ok((l, sig))
}

/// The release public key. Zeroed here: this repository does not hold the private half,
/// and shipping a real key alongside a placeholder signature would make `verify` look
/// like it worked. With a zero key every signature fails closed, which is the correct
/// behaviour for a build that cannot check anything.
pub const RELEASE_PUBLIC_KEY: [u8; 32] = [0u8; 32];

#[cfg(feature = "stats")]
pub fn verify(text: &str, today: &str, public_key: &[u8; 32]) -> Result<License, LicenseError> {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    let (l, sig) = parse(text)?;
    let vk = VerifyingKey::from_bytes(public_key).map_err(|_| LicenseError::UnknownKey)?;
    let sig: [u8; 64] = sig.try_into().map_err(|_| LicenseError::Tampered)?;
    vk.verify(payload(&l).as_bytes(), &Signature::from_bytes(&sig))
        .map_err(|_| LicenseError::Tampered)?;
    // Expiry is checked AFTER the signature: an expired token whose signature does not
    // verify is a forgery, and reporting it as merely expired would be misleading.
    if l.expires.as_str() < today {
        return Err(LicenseError::Expired {
            on: l.expires.clone(),
        });
    }
    Ok(l)
}

#[cfg(not(feature = "stats"))]
pub fn verify(_text: &str, _today: &str, _public_key: &[u8; 32]) -> Result<License, LicenseError> {
    Err(LicenseError::Malformed(
        "this build has no signature support; rebuild with --features stats".into(),
    ))
}

/// Seal a corpus snapshot with the content key.
#[cfg(feature = "stats")]
pub fn seal(plaintext: &[u8], key_hex: &str, nonce: &[u8; 24]) -> Result<Vec<u8>, String> {
    use chacha20poly1305::aead::{Aead, KeyInit};
    use chacha20poly1305::XChaCha20Poly1305;
    let key = hex_decode(key_hex)?;
    let key: [u8; 32] = key.try_into().map_err(|_| "content key must be 32 bytes")?;
    let c = XChaCha20Poly1305::new(&key.into());
    let mut out = nonce.to_vec();
    out.extend(
        c.encrypt(nonce.into(), plaintext)
            .map_err(|e| e.to_string())?,
    );
    Ok(out)
}

/// Open a sealed snapshot. A tampered or truncated one fails whole, never partially.
#[cfg(feature = "stats")]
pub fn unseal(sealed: &[u8], key_hex: &str) -> Result<Vec<u8>, String> {
    use chacha20poly1305::aead::{Aead, KeyInit};
    use chacha20poly1305::XChaCha20Poly1305;
    if sealed.len() < 24 {
        return Err("sealed corpus is truncated".into());
    }
    let key = hex_decode(key_hex)?;
    let key: [u8; 32] = key.try_into().map_err(|_| "content key must be 32 bytes")?;
    let c = XChaCha20Poly1305::new(&key.into());
    let (nonce, body) = sealed.split_at(24);
    let nonce: [u8; 24] = nonce.try_into().map_err(|_| "bad nonce")?;
    c.decrypt(&nonce.into(), body)
        .map_err(|_| "the sealed corpus failed its integrity check".to_string())
}

/// Without the `stats` feature there is no AEAD in this build, so a sealed corpus cannot
/// be opened at all. Saying which build is missing is more useful than "failed".
#[cfg(not(feature = "stats"))]
pub fn seal(_plaintext: &[u8], _key_hex: &str, _nonce: &[u8; 24]) -> Result<Vec<u8>, String> {
    Err("this build has no corpus support; rebuild with --features stats".into())
}

#[cfg(not(feature = "stats"))]
pub fn unseal(_sealed: &[u8], _key_hex: &str) -> Result<Vec<u8>, String> {
    Err("this build has no corpus support; rebuild with --features stats".into())
}

/// What `tile license status` reports.
pub fn status(today: &str) -> Result<License, LicenseError> {
    let p = path();
    if !p.exists() {
        return Err(LicenseError::Absent);
    }
    let text = std::fs::read_to_string(&p).map_err(|e| LicenseError::Malformed(e.to_string()))?;
    verify(&text, today, &RELEASE_PUBLIC_KEY)
}

#[cfg(all(test, feature = "stats"))]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn signed(l: &License, sk: &SigningKey) -> String {
        let sig = sk.sign(payload(l).as_bytes());
        format!("{}signature={}\n", payload(l), hex_encode(&sig.to_bytes()))
    }

    fn a_license() -> License {
        License {
            subject: "a team".into(),
            expires: "2030-01-01".into(),
            features: vec!["corpus".into()],
            content_key: hex_encode(&[7u8; 32]),
        }
    }

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[42u8; 32])
    }

    #[test]
    fn a_properly_signed_license_verifies_offline() {
        let sk = key();
        let text = signed(&a_license(), &sk);
        let got = verify(&text, "2026-09-01", &sk.verifying_key().to_bytes()).unwrap();
        assert_eq!(got.subject, "a team");
        assert_eq!(got.features, vec!["corpus".to_string()]);
    }

    #[test]
    fn a_token_signed_by_another_key_is_refused() {
        let text = signed(&a_license(), &key());
        let other = SigningKey::from_bytes(&[9u8; 32]);
        let e = verify(&text, "2026-09-01", &other.verifying_key().to_bytes()).unwrap_err();
        assert_eq!(e, LicenseError::Tampered);
    }

    #[test]
    fn a_tampered_payload_is_refused() {
        // Raise the feature set without re-signing: the classic edit.
        let text = signed(&a_license(), &key()).replace("features=corpus", "features=corpus,all");
        let e = verify(&text, "2026-09-01", &key().verifying_key().to_bytes()).unwrap_err();
        assert_eq!(e, LicenseError::Tampered);
    }

    #[test]
    fn an_expired_token_says_when_it_expired() {
        let mut l = a_license();
        l.expires = "2020-01-01".into();
        let text = signed(&l, &key());
        let e = verify(&text, "2026-09-01", &key().verifying_key().to_bytes()).unwrap_err();
        assert_eq!(
            e,
            LicenseError::Expired {
                on: "2020-01-01".into()
            }
        );
    }

    #[test]
    fn expiry_is_checked_after_the_signature() {
        // An expired token whose signature does not verify is a FORGERY. Reporting it as
        // merely expired would tell the holder to renew something that was never valid.
        let mut l = a_license();
        l.expires = "2020-01-01".into();
        let text = signed(&l, &key());
        let other = SigningKey::from_bytes(&[9u8; 32]);
        let e = verify(&text, "2026-09-01", &other.verifying_key().to_bytes()).unwrap_err();
        assert_eq!(
            e,
            LicenseError::Tampered,
            "a forgery must not be reported as expired"
        );
    }

    #[test]
    fn the_shipped_public_key_fails_closed() {
        // This repository does not hold the private half. A build that cannot check
        // anything must reject everything, not accept everything.
        let text = signed(&a_license(), &key());
        assert!(verify(&text, "2026-09-01", &RELEASE_PUBLIC_KEY).is_err());
    }

    #[test]
    fn a_sealed_corpus_round_trips_and_is_not_readable_as_text() {
        let plain = b"softmax\tmetal\timplemented\t0.31\tmeasured\t2026-08-01\n";
        let k = hex_encode(&[3u8; 32]);
        let sealed = seal(plain, &k, &[1u8; 24]).unwrap();
        assert!(
            !String::from_utf8_lossy(&sealed).contains("softmax"),
            "the corpus is readable without the key"
        );
        assert_eq!(unseal(&sealed, &k).unwrap(), plain);
    }

    #[test]
    fn a_tampered_seal_fails_whole_rather_than_opening_partially() {
        let plain = b"a record";
        let k = hex_encode(&[3u8; 32]);
        let mut sealed = seal(plain, &k, &[1u8; 24]).unwrap();
        let last = sealed.len() - 1;
        sealed[last] ^= 0xff;
        let e = unseal(&sealed, &k).unwrap_err();
        assert!(e.contains("integrity check"), "{e}");
    }

    #[test]
    fn the_wrong_key_opens_nothing() {
        let sealed = seal(b"x", &hex_encode(&[3u8; 32]), &[1u8; 24]).unwrap();
        assert!(unseal(&sealed, &hex_encode(&[4u8; 32])).is_err());
    }

    #[test]
    fn a_truncated_seal_is_refused_rather_than_indexed_past_its_end() {
        assert!(unseal(&[0u8; 5], &hex_encode(&[3u8; 32])).is_err());
    }

    #[test]
    fn an_unknown_field_in_a_license_is_refused() {
        let e =
            parse("subject=a\nexpires=2030-01-01\nkey=aa\nrole=admin\nsignature=00\n").unwrap_err();
        assert!(matches!(e, LicenseError::Malformed(_)), "{e:?}");
    }
}
