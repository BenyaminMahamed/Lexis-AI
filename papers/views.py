import json
import logging

from django.conf import settings
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .models import Paper, Chunk
from .pipeline import process_paper, answer_question

logger = logging.getLogger(__name__)

VALID_MODES = {'qa', 'summarise', 'critique', 'compare'}


def index(request):
    """Homepage — upload form + list of papers."""
    papers = Paper.objects.all()
    return render(request, 'papers/index.html', {
        'papers': papers,
        'has_api_key': bool(settings.GEMINI_API_KEY),
    })


def upload_paper(request):
    if request.method != 'POST':
        return redirect('index')

    uploaded_file = request.FILES.get('pdf_file')
    if not uploaded_file:
        return redirect('index')

    paper = Paper.objects.create(uploaded_file=uploaded_file)

    try:
        chunk_count = process_paper(paper)
        logger.info(f"Processed paper {paper.id} — {chunk_count} chunks")
    except Exception as e:
        logger.error(f"Processing failed for paper {paper.id}: {e}")
        paper.delete()
        return render(request, 'papers/index.html', {
            'papers': Paper.objects.all(),
            'error': f"Failed to process PDF: {str(e)}"
        })

    return redirect('paper_detail', pk=paper.pk)


def paper_detail(request, pk):
    paper = get_object_or_404(Paper, pk=pk)
    all_papers = Paper.objects.filter(processed=True)
    return render(request, 'papers/detail.html', {
        'paper': paper,
        'all_papers': all_papers,
    })


@require_POST
def ask(request):
    """AJAX endpoint for Q&A."""
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        paper_ids = data.get('paper_ids', [])
        mode = data.get('mode', 'qa')

        if not question:
            return JsonResponse({'error': 'No question provided.'}, status=400)
        if not paper_ids:
            return JsonResponse({'error': 'No papers selected.'}, status=400)
        if mode not in VALID_MODES:
            return JsonResponse({'error': f'Invalid mode. Must be one of: {", ".join(VALID_MODES)}'}, status=400)

        result = answer_question(question, paper_ids=paper_ids, mode=mode)
        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body.'}, status=400)
    except Exception as e:
        logger.error(f"Ask endpoint error: {e}")
        return JsonResponse({'error': 'An unexpected error occurred. Please try again.'}, status=500)


def delete_paper(request, pk):
    paper = get_object_or_404(Paper, pk=pk)
    if request.method == 'POST':
        paper.delete()
    return redirect('index')