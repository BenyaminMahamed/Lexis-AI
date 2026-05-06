from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import Paper, Chunk
from .serializers import PaperSerializer, QueryRequestSerializer
from .pipeline import answer_question

class PaperViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = PaperSerializer

    def get_queryset(self):
        return Paper.objects.all().order_by('-created_at')

    @action(detail=True, methods=['post'], url_path='ask')
    def ask(self, request, pk=None):
        paper = get_object_or_404(Paper, pk=pk)
        serializer = QueryRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        data = serializer.validated_data
        paper_ids = data.get('paper_ids') or [paper.id]
        try:
            result = answer_question(data['question'], paper_ids=paper_ids, mode=data['mode'])
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(result, status=status.HTTP_200_OK)

    @action(detail=True, methods=['get'], url_path='chunks')
    def chunks(self, request, pk=None):
        paper = get_object_or_404(Paper, pk=pk)
        chunks = Chunk.objects.filter(paper=paper).order_by('chunk_index')
        data = [
            {
                'chunk_index': c.chunk_index,
                'page': c.page_number,
                'text_preview': c.text[:200],
            }
            for c in chunks
        ]
        return Response({'paper_id': paper.id, 'chunk_count': len(data), 'chunks': data})
