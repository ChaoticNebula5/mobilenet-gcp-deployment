terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
  required_version = ">= 1.0"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "cloud_run_api" {
  service = "run.googleapis.com"
  
  disable_dependent_services = true
}

resource "google_project_service" "container_registry_api" {
  service = "containerregistry.googleapis.com"
  
  disable_dependent_services = true
}

resource "google_project_service" "cloud_build_api" {
  service = "cloudbuild.googleapis.com"
  
  disable_dependent_services = true
}

resource "google_cloud_run_service" "mobilenet_classifier" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      containers {
        image = var.image_url
        
        ports {
          container_port = 8000
        }
        
        resources {
          limits = {
            cpu    = "1000m"
            memory = "1Gi"
          }
        }
        
        env {
          name  = "PORT"
          value = "8000"
        }
      }
      
      # Set timeout for longer model loading
      timeout_seconds = 300
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "10"
        "run.googleapis.com/cpu-throttling" = "false"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_project_service.cloud_run_api]
}

resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_service.mobilenet_classifier.name
  location = google_cloud_run_service.mobilenet_classifier.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}