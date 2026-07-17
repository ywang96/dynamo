// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom Resource Definition for DynamoWorkerMetadata
//!
//! This module defines the Rust types for the DynamoWorkerMetadata CRD,
//! which stores discovery metadata for Dynamo worker pods in Kubernetes.
//!
//! The CRD schema is defined in the Helm chart at:
//! `deploy/helm/charts/crds/templates/nvidia.com_dynamoworkermetadatas.yaml`

use std::future::Future;
use std::time::Duration;

use anyhow::Result;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference;
use kube::{
    Api, Client as KubeClient, CustomResource,
    api::{Patch, PatchParams},
};
use serde::{Deserialize, Serialize};

use crate::discovery::DiscoveryMetadata;

/// Field manager name for server-side apply - identifies this client as the owner of fields it sets
const FIELD_MANAGER: &str = "dynamo-worker";

/// Total wall-clock budget for retrying a CR apply. Kept well under the graceful
/// shutdown grace period (`DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS`, default 5s)
/// so retries always complete before the worker proceeds to drain/cleanup.
const APPLY_RETRY_BUDGET: Duration = Duration::from_secs(10);
const APPLY_INITIAL_BACKOFF: Duration = Duration::from_millis(250);
const APPLY_MAX_BACKOFF: Duration = Duration::from_secs(2);

/// Bounded retry policy for [`apply_cr`].
#[derive(Clone, Copy, Debug)]
struct RetryPolicy {
    budget: Duration,
    initial_backoff: Duration,
    max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            budget: APPLY_RETRY_BUDGET,
            initial_backoff: APPLY_INITIAL_BACKOFF,
            max_backoff: APPLY_MAX_BACKOFF,
        }
    }
}

/// Double the backoff, capped at `max`. Saturating so it never panics on overflow.
fn next_backoff(current: Duration, max: Duration) -> Duration {
    current.saturating_mul(2).min(max)
}

/// True if any error in the chain is a Kubernetes API 401/403.
///
/// Relies on the typed `kube::Error` being preserved in the chain. [`apply_cr_once`]
/// uses `anyhow::Error::new(e).context(..)` (NOT `anyhow!("{}", e)`) for exactly
/// this reason — string-formatting the kube error would make it undetectable here.
fn is_auth_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<kube::Error>()
            .is_some_and(|k| matches!(k, kube::Error::Api(resp) if matches!(resp.code, 401 | 403)))
    })
}

/// Spec for DynamoWorkerMetadata custom resource
/// The `data` field stores the serialized `DiscoveryMetadata` as a JSON blob.
#[derive(CustomResource, Clone, Debug, Deserialize, Serialize)]
#[kube(
    group = "nvidia.com",
    version = "v1alpha1",
    kind = "DynamoWorkerMetadata",
    namespaced,
    schema = "disabled"
)]
pub struct DynamoWorkerMetadataSpec {
    /// Raw JSON blob containing the DiscoveryMetadata
    pub data: serde_json::Value,
}

impl DynamoWorkerMetadataSpec {
    pub fn new(data: serde_json::Value) -> Self {
        Self { data }
    }
}

/// Build a DynamoWorkerMetadata CR with owner reference set to the pod
/// # Arguments
/// * `cr_name` - Name of the CR (from KubeDiscoveryTarget::cr_name)
/// * `pod_name` - Name of the pod (used in owner reference)
/// * `pod_uid` - UID of the pod (for owner reference - enables garbage collection)
/// * `metadata` - The DiscoveryMetadata to serialize into the CR's data field
///
/// # Returns
/// A `DynamoWorkerMetadata` CR ready to be applied to the cluster
pub fn build_cr(
    cr_name: &str,
    pod_name: &str,
    pod_uid: &str,
    metadata: &DiscoveryMetadata,
) -> Result<DynamoWorkerMetadata> {
    let data = serde_json::to_value(metadata)?;
    let spec = DynamoWorkerMetadataSpec::new(data);
    let mut cr = DynamoWorkerMetadata::new(cr_name, spec);

    // Set owner reference to the pod for automatic garbage collection
    cr.metadata.owner_references = Some(vec![OwnerReference {
        api_version: "v1".to_string(),
        kind: "Pod".to_string(),
        name: pod_name.to_string(),
        uid: pod_uid.to_string(),
        // Mark pod as the controlling owner - CR will be garbage collected when pod is deleted.
        // In container mode multiple CRs may share one pod; only one can be controller.
        controller: Some(cr_name == pod_name),
        // Don't block pod deletion - allow CR cleanup to happen asynchronously
        block_owner_deletion: Some(false),
    }]);

    Ok(cr)
}

/// Single CR apply attempt using server-side apply.
///
/// Preserves the typed `kube::Error` in the returned error chain (via
/// `anyhow::Error::new(..).context(..)`) so [`is_auth_error`] can classify
/// failures. Do NOT change this to `anyhow!("...: {}", e)` — that stringifies
/// the kube error and breaks 401/403 detection in the retry loop.
async fn apply_cr_once(
    kube_client: &KubeClient,
    namespace: &str,
    cr: &DynamoWorkerMetadata,
) -> Result<()> {
    let api: Api<DynamoWorkerMetadata> = Api::namespaced(kube_client.clone(), namespace);

    let cr_name = cr
        .metadata
        .name
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("CR must have a name"))?;

    // force() allows us to take ownership of this field even if another controller owns it
    // in practice the CR will only have one writer (the pod owner)
    let params = PatchParams::apply(FIELD_MANAGER).force();

    api.patch(cr_name, &params, &Patch::Apply(cr))
        .await
        .map_err(|e| anyhow::Error::new(e).context("Failed to apply DynamoWorkerMetadata CR"))?;

    tracing::debug!(
        "Applied DynamoWorkerMetadata CR: name={}, namespace={}",
        cr_name,
        namespace
    );

    Ok(())
}

/// Retry an idempotent apply, rebuilding the client on auth (401/403) errors.
///
/// Generic over the client type `C` so the loop is unit-testable with a fake.
/// In production `C = KubeClient` and `make_client = KubeClient::try_default`.
async fn retry_loop<C, F, Fut, M, MFut>(
    mut client: C,
    mut attempt: F,
    mut make_client: M,
    policy: RetryPolicy,
) -> Result<()>
where
    C: Clone,
    F: FnMut(C) -> Fut,
    Fut: Future<Output = Result<()>>,
    M: FnMut() -> MFut,
    MFut: Future<Output = Result<C>>,
{
    let deadline = tokio::time::Instant::now() + policy.budget;
    let mut backoff = policy.initial_backoff;
    let mut attempt_no: u32 = 0;

    loop {
        attempt_no += 1;
        match attempt(client.clone()).await {
            Ok(()) => {
                if attempt_no > 1 {
                    tracing::info!(attempt = attempt_no, "apply_cr succeeded after retry");
                }
                return Ok(());
            }
            Err(e) => {
                if tokio::time::Instant::now() >= deadline {
                    return Err(e.context(format!(
                        "apply_cr exhausted {:?} retry budget after {attempt_no} attempts",
                        policy.budget
                    )));
                }

                let auth = is_auth_error(&e);
                tracing::warn!(
                    attempt = attempt_no,
                    auth_failure = auth,
                    error = %e,
                    "apply_cr attempt failed; retrying after {:?}",
                    backoff
                );

                if auth {
                    // Cached service-account token is stale. Rebuild the client to
                    // force a fresh read from the projected token file.
                    match make_client().await {
                        Ok(c) => {
                            tracing::info!("Rebuilt kube client to force token refresh");
                            client = c;
                        }
                        Err(rebuild_err) => {
                            tracing::warn!(
                                error = %rebuild_err,
                                "Failed to rebuild kube client; retrying with existing client"
                            );
                        }
                    }
                }

                tokio::time::sleep(backoff).await;
                backoff = next_backoff(backoff, policy.max_backoff);
            }
        }
    }
}

/// Apply (create or update) a DynamoWorkerMetadata CR using server-side apply,
/// with bounded retry and client refresh on auth failures.
///
/// This function uses Kubernetes server-side apply which:
/// - Creates the CR if it doesn't exist
/// - Updates the CR if it does exist
/// - Is idempotent and safe to call multiple times
///
/// On 401/403 the cached service-account token in the kube client is stale
/// (kubelet rotated the projected token; kube-rs caches it for up to 60s, and
/// stops rotating once the pod is terminating). We rebuild the client to force a
/// fresh token read. Bounded by [`RetryPolicy`] so it always finishes inside the
/// shutdown grace period.
///
/// # Arguments
/// * `kube_client` - Kubernetes client
/// * `namespace` - Namespace to create/update the CR in
/// * `cr` - The DynamoWorkerMetadata CR to apply
pub async fn apply_cr(
    kube_client: &KubeClient,
    namespace: &str,
    cr: &DynamoWorkerMetadata,
) -> Result<()> {
    retry_loop(
        kube_client.clone(),
        |client| async move { apply_cr_once(&client, namespace, cr).await },
        || async {
            KubeClient::try_default()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to rebuild kube client: {e}"))
        },
        RetryPolicy::default(),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use kube::Resource;

    #[test]
    fn test_crd_metadata() {
        // Verify the CRD metadata is correct
        assert_eq!(DynamoWorkerMetadata::group(&()), "nvidia.com");
        assert_eq!(DynamoWorkerMetadata::version(&()), "v1alpha1");
        assert_eq!(DynamoWorkerMetadata::kind(&()), "DynamoWorkerMetadata");
        assert_eq!(DynamoWorkerMetadata::plural(&()), "dynamoworkermetadatas");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let data = serde_json::json!({
            "endpoints": {
                "ns/comp/ep": {
                    "type": "Endpoint",
                    "namespace": "ns",
                    "component": "comp",
                    "endpoint": "ep",
                    "instance_id": 12345,
                    "transport": { "Nats": "nats://localhost:4222" }
                }
            },
            "model_cards": {}
        });

        let spec = DynamoWorkerMetadataSpec::new(data.clone());

        let cr = DynamoWorkerMetadata::new("test-pod", spec);

        let json = serde_json::to_string(&cr).expect("Failed to serialize CR");

        let deserialized: DynamoWorkerMetadata =
            serde_json::from_str(&json).expect("Failed to deserialize CR");

        assert_eq!(deserialized.spec.data, data);
    }

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Mirror exactly how `apply_cr_once` wraps the kube error: anyhow context
    /// over the typed `kube::Error`, so the chain still contains `kube::Error`.
    fn api_error(code: u16) -> anyhow::Error {
        let kube_err = kube::Error::Api(kube::core::ErrorResponse {
            status: "Failure".to_string(),
            message: "Unauthorized".to_string(),
            reason: "Unauthorized".to_string(),
            code,
        });
        anyhow::Error::new(kube_err).context("Failed to apply DynamoWorkerMetadata CR")
    }

    #[test]
    fn is_auth_error_detects_401() {
        assert!(super::is_auth_error(&api_error(401)));
    }

    #[test]
    fn is_auth_error_detects_403() {
        assert!(super::is_auth_error(&api_error(403)));
    }

    #[test]
    fn is_auth_error_ignores_404() {
        assert!(!super::is_auth_error(&api_error(404)));
    }

    #[test]
    fn is_auth_error_ignores_non_kube_error() {
        let err = anyhow::anyhow!("some unrelated failure").context("outer");
        assert!(!super::is_auth_error(&err));
    }

    #[test]
    fn next_backoff_doubles_then_caps() {
        let max = Duration::from_secs(2);
        assert_eq!(
            super::next_backoff(Duration::from_millis(250), max),
            Duration::from_millis(500)
        );
        assert_eq!(
            super::next_backoff(Duration::from_millis(500), max),
            Duration::from_secs(1)
        );
        assert_eq!(
            super::next_backoff(Duration::from_secs(1), max),
            Duration::from_secs(2)
        );
        // caps at max
        assert_eq!(
            super::next_backoff(Duration::from_secs(2), max),
            Duration::from_secs(2)
        );
        assert_eq!(
            super::next_backoff(Duration::from_secs(10), max),
            Duration::from_secs(2)
        );
    }

    #[test]
    fn retry_policy_default_is_bounded_under_grace_period() {
        let p = super::RetryPolicy::default();
        // Must finish well under the grace period so retries never bleed into drain.
        assert!(p.budget <= Duration::from_secs(10));
        assert!(p.initial_backoff <= p.max_backoff);
    }

    #[tokio::test(start_paused = true)]
    async fn retry_loop_rebuilds_client_on_auth_error_then_succeeds() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let rebuilds = Arc::new(AtomicUsize::new(0));

        let a = attempts.clone();
        let r = rebuilds.clone();
        let result = super::retry_loop(
            0u32, // fake "client": a version tag
            move |_client| {
                let a = a.clone();
                async move {
                    let n = a.fetch_add(1, Ordering::SeqCst);
                    if n < 2 {
                        Err(api_error(401)) // typed kube 401, like apply_cr_once produces
                    } else {
                        Ok(())
                    }
                }
            },
            move || {
                let r = r.clone();
                async move { Ok(r.fetch_add(1, Ordering::SeqCst) as u32 + 1) }
            },
            super::RetryPolicy::default(),
        )
        .await;

        assert!(result.is_ok(), "expected success after retries: {result:?}");
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            3,
            "two failures then a success"
        );
        assert_eq!(
            rebuilds.load(Ordering::SeqCst),
            2,
            "client rebuilt on each auth failure"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn retry_loop_exhausts_budget_on_persistent_failure() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let rebuilds = Arc::new(AtomicUsize::new(0));

        let a = attempts.clone();
        let r = rebuilds.clone();
        let result = super::retry_loop(
            0u32,
            move |_client| {
                let a = a.clone();
                async move {
                    a.fetch_add(1, Ordering::SeqCst);
                    Err(api_error(500)) // non-auth, never recovers
                }
            },
            move || {
                let r = r.clone();
                async move {
                    r.fetch_add(1, Ordering::SeqCst);
                    Ok(1u32)
                }
            },
            super::RetryPolicy::default(),
        )
        .await;

        assert!(result.is_err(), "persistent failure must surface an error");
        assert!(
            attempts.load(Ordering::SeqCst) > 1,
            "should retry at least once"
        );
        assert_eq!(
            rebuilds.load(Ordering::SeqCst),
            0,
            "no client rebuild for non-auth errors"
        );
    }
}
