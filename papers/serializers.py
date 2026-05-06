from rest_framework import serializers
from .models import Paper, Chunk


class PaperSerializer(serializers.ModelSerializer):
    class Meta:
        model = Paper
        fields = ['id', 'title', 'uploaded_at', 'chunk_count', 'processed']
        read_only_fields = fields


class QueryRequestSerializer(serializers.Serializer):
    MODES = [('qa', 'Q&A'), ('summarise', 'Summarise'), ('critique', 'Critique'), ('compare', 'Compare')]
    question = serializers.CharField(max_length=2000)
    mode = serializers.ChoiceField(choices=MODES, default='qa')
    paper_ids = serializers.ListField(child=serializers.IntegerField(), required=False, default=list)


class SourceChunkSerializer(serializers.Serializer):
    text = serializers.CharField()
    page = serializers.IntegerField()
    paper_id = serializers.IntegerField()
    paper_title = serializers.CharField()


class QueryResponseSerializer(serializers.Serializer):
    answer = serializers.CharField()
    mode = serializers.CharField()
    sources = SourceChunkSerializer(many=True)
    paper_id = serializers.IntegerField()
    paper_title = serializers.CharField()
