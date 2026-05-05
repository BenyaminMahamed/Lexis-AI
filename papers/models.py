from django.db import models


class Paper(models.Model):
    title = models.CharField(max_length=500, blank=True)
    uploaded_file = models.FileField(upload_to='uploads/')
    created_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    chunk_count = models.IntegerField(default=0)

    def __str__(self):
        return self.title or f"Paper #{self.pk}"

    class Meta:
        ordering = ['-created_at']


class Chunk(models.Model):
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE, related_name='chunks')
    text = models.TextField()
    chunk_index = models.IntegerField()
    page_number = models.IntegerField(default=0)
    # FAISS index position stored here so we can map results back to source
    faiss_id = models.IntegerField(default=-1)

    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.paper}"

    class Meta:
        ordering = ['paper', 'chunk_index']