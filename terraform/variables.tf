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
  default     = "gcr.io/PROJECT_ID/mobilenet-classifier:latest"
}