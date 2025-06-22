variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "The name of the Cloud Run service"
  type        = string
  default     = "mobilenet-classifier"
}

variable "image_url" {
  description = "The container image URL"
  type        = string
  default     = "asia-south1-docker.pkg.dev/PROJECT_ID/mobilenet-classifier/mobilenet-classifier:latest"
}