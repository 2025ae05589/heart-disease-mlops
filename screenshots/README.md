# Screenshots

Screenshots of the run of record, grouped by assignment task. Screenshots 00 to 23 have been taken on the BITS lab machine, 24 to 27 on GitHub and 28 to 37 on Google Cloud. The section numbers refer to the report.

## Setup and the lab run (report section 3)

- [00_lab_session.jpeg](00_lab_session.jpeg) - lab session (BITS virtual lab)
- [01_clone_and_setup.jpeg](01_clone_and_setup.jpeg) - clone and setup
- [01b_repo_files.png](01b_repo_files.png) - repository files
- [02a_lint_and_tests.jpeg](02a_lint_and_tests.jpeg) - lint and tests
- [02b_lint_and_tests.jpeg](02b_lint_and_tests.jpeg) - 94 tests passed
- [03_data_download_check.jpeg](03_data_download_check.jpeg) - data download and hash check

## Task 1 - Data acquisition and EDA (report section 4)

- [04_eda_target_distribution.jpeg](04_eda_target_distribution.jpeg) - EDA: class balance
- [05_eda_correlation.jpeg](05_eda_correlation.jpeg) - EDA: correlation

## Task 2 - Feature engineering and model development (report section 5)

- [06_model_comparison.jpeg](06_model_comparison.jpeg) - model comparison
- [07_test_results.jpeg](07_test_results.jpeg) - test-set plots
- [07a_test_metrics.jpeg](07a_test_metrics.jpeg) - test metrics table

## Task 3 - Experiment tracking (report section 6)

- [20_mlflow_experiment_runs.jpeg](20_mlflow_experiment_runs.jpeg) - MLflow runs
- [21a_mlflow_run_metrics.jpeg](21a_mlflow_run_metrics.jpeg) - champion run metrics
- [21b_mlflow_run_parameters.jpeg](21b_mlflow_run_parameters.jpeg) - champion run parameters
- [21c_mlflow_run_logged_model.jpeg](21c_mlflow_run_logged_model.jpeg) - logged model
- [21d_mlflow_run2_metrics.jpeg](21d_mlflow_run2_metrics.jpeg) - run 2 metrics
- [21e_mlflow_run2_parameters.jpeg](21e_mlflow_run2_parameters.jpeg) - run 2 parameters
- [22a_mlflow_cv_confusion_matrix.jpeg](22a_mlflow_cv_confusion_matrix.jpeg) - CV confusion matrix
- [22b_mlflow_cv_roc_curve.jpeg](22b_mlflow_cv_roc_curve.jpeg) - CV ROC curve
- [22c_mlflow_test_confusion_matrix.jpeg](22c_mlflow_test_confusion_matrix.jpeg) - test confusion matrix
- [22d_mlflow_test_roc_curve.jpeg](22d_mlflow_test_roc_curve.jpeg) - test ROC curve
- [23a_mlflow_registry_aliases.jpeg](23a_mlflow_registry_aliases.jpeg) - registry aliases
- [23b_mlflow_registry_version_2.jpeg](23b_mlflow_registry_version_2.jpeg) - registry version 2
- [23c_mlflow_registry_version_1.jpeg](23c_mlflow_registry_version_1.jpeg) - registry version 1
- [23d_mlflow_registry_versions.jpeg](23d_mlflow_registry_versions.jpeg) - registry versions

## Task 4 - Model packaging and reproducibility (report section 7)

- [08_pipeline_run_1.jpeg](08_pipeline_run_1.jpeg) - pipeline run 1
- [19_pipeline_run_2_gate2.jpeg](19_pipeline_run_2_gate2.jpeg) - pipeline run 2, gate 2
- [15_notebook03_api_compare.jpeg](15_notebook03_api_compare.jpeg) - notebook 03 vs API

## Task 5 - CI/CD pipeline and automated testing (report section 8)

- [24_ci_run_jobs.jpeg](24_ci_run_jobs.jpeg) - CI run, six jobs
- [24a_ci_artifacts.jpeg](24a_ci_artifacts.jpeg) - CI artifacts
- [24b_ci_prepare_data.jpeg](24b_ci_prepare_data.jpeg) - CI prepare-data
- [24c_ci_train.jpeg](24c_ci_train.jpeg) - CI training
- [24d_ci_evaluate.jpeg](24d_ci_evaluate.jpeg) - CI evaluation, gate 1
- [24e_ci_register_export.jpeg](24e_ci_register_export.jpeg) - CI register, export, check
- [24f_ci_build.jpeg](24f_ci_build.jpeg) - CI image build
- [24g_ci_container_test.jpeg](24g_ci_container_test.jpeg) - CI container test
- [24h_ci_container_metrics.jpeg](24h_ci_container_metrics.jpeg) - CI container metrics
- [24i_ci_run_graph.jpeg](24i_ci_run_graph.jpeg) - CI job graph
- [25_ci_test_log.jpeg](25_ci_test_log.jpeg) - CI 94 passed
- [25a_ci_lint.jpeg](25a_ci_lint.jpeg) - CI lint
- [25b_ci_tests_start.jpeg](25b_ci_tests_start.jpeg) - CI tests start

## Task 6 - Model containerization (report section 9)

- [10_docker_build.jpeg](10_docker_build.jpeg) - docker build
- [12_api_curl.jpeg](12_api_curl.jpeg) - curl requests
- [12a_api_curl_batch_metrics.jpeg](12a_api_curl_batch_metrics.jpeg) - batch and metrics
- [13_api_docs_predict.jpeg](13_api_docs_predict.jpeg) - /docs predict

## Task 7 - Production deployment (report section 10)

- [10a_minikube_start.jpeg](10a_minikube_start.jpeg) - Minikube start
- [11_minikube_pods_service.jpeg](11_minikube_pods_service.jpeg) - Minikube pods and service
- [26_ci_deploy_log.jpeg](26_ci_deploy_log.jpeg) - CI deploy to GKE
- [26a_ci_deploy_public_address.jpeg](26a_ci_deploy_public_address.jpeg) - CI public address check
- [27_ghcr_package.jpeg](27_ghcr_package.jpeg) - GHCR package
- [28_gke_api_docs.jpeg](28_gke_api_docs.jpeg) - GKE /docs predict
- [28a_gke_api_curl.jpeg](28a_gke_api_curl.jpeg) - GKE curl requests
- [28b_gke_api_batch.jpeg](28b_gke_api_batch.jpeg) - GKE batch
- [29_gke_model_info.jpeg](29_gke_model_info.jpeg) - GKE model-info
- [30_gke_cluster.jpeg](30_gke_cluster.jpeg) - GKE cluster
- [31_gke_workloads.jpeg](31_gke_workloads.jpeg) - GKE workloads
- [32_gke_services.jpeg](32_gke_services.jpeg) - GKE services
- [33_artifact_registry.jpeg](33_artifact_registry.jpeg) - Artifact Registry images
- [33a_artifact_registry_repo.jpeg](33a_artifact_registry_repo.jpeg) - Artifact Registry repo
- [34_gke_kubectl.jpeg](34_gke_kubectl.jpeg) - Cloud Shell kubectl, helm, logs
- [34a_gke_kubectl_top.jpeg](34a_gke_kubectl_top.jpeg) - Cloud Shell kubectl

## Task 8 - Monitoring and logging (report section 11)

- [14_api_logs.jpeg](14_api_logs.jpeg) - API logs (Minikube)
- [16_monitoring_pods.jpeg](16_monitoring_pods.jpeg) - monitoring pods
- [17_grafana_dashboard.jpeg](17_grafana_dashboard.jpeg) - Grafana (Minikube)
- [18_api_metrics_endpoint.jpeg](18_api_metrics_endpoint.jpeg) - /metrics endpoint
- [28c_gke_api_metrics.jpeg](28c_gke_api_metrics.jpeg) - GKE metrics
- [35_gke_grafana_dashboard.jpeg](35_gke_grafana_dashboard.jpeg) - Grafana (GKE)
- [36_gke_prometheus_targets.jpeg](36_gke_prometheus_targets.jpeg) - Prometheus targets (GKE)
- [37_gke_cloud_logging.jpeg](37_gke_cloud_logging.jpeg) - Cloud Logging
- [37a_gke_log_entry.jpeg](37a_gke_log_entry.jpeg) - Cloud Logging entry
