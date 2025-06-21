output "service_url" {
  description = "The URL of the deployed Cloud Run service"
  value       = google_cloud_run_service.mobilenet_classifier.status[0].url
}

output "service_name" {
  description = "The name of the Cloud Run service"
  value       = google_cloud_run_service.mobilenet_classifier.name
}

output "project_id" {
  description = "The GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "The GCP region"
  value       = var.region
}