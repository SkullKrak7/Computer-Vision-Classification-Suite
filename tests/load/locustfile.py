"""Load testing with Locust"""

from io import BytesIO

from locust import HttpUser, between, task
from PIL import Image


class CVClassificationUser(HttpUser):
    """Simulated user for load testing"""

    wait_time = between(1, 3)

    def on_start(self):
        """Setup - create test image"""
        img = Image.new("RGB", (100, 100), color="red")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        self.test_image = buf.getvalue()

    @task(3)
    def health_check(self):
        """Test health endpoint"""
        self.client.get("/health")

    @task(1)
    def metrics_check(self):
        """Test metrics endpoint"""
        self.client.get("/metrics")

    @task(5)
    def predict_image(self):
        """Test prediction endpoint"""
        files = {"file": ("test.jpg", BytesIO(self.test_image), "image/jpeg")}
        self.client.post("/v1/inference/predict", files=files)
