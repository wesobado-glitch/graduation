# make the three endpoints like what exist in app.py but with django rest framework

# pyrefly: ignore [missing-import]
import os
import torch
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK, HTTP_500_INTERNAL_SERVER_ERROR
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated

from recommender.sasrec.domain import RecommendRequest, RecommendResponse, CustomerRecommendRequest, CustomerRecommendResponse
from recommender.sasrec.infrastructure.model_store import load_checkpoint
from recommender.sasrec.infrastructure.db_connection import get_customer_last_items, get_trending_items
from recommender.sasrec.application_services.recommender import get_recommendations


MODEL_PT = "recommender/model.pt"
CHECKPOINT_JSON = "recommender/checkpoint.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy-load: only initialize the model on first request, not during migrate/collectstatic
_state = None

def _get_state():
    global _state
    if _state is None:
        _state = load_checkpoint(model_pt=MODEL_PT, checkpoint_json=CHECKPOINT_JSON)
    return _state

# POST /recommend item id's to user 

@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def recommend(request):
    print(request.data)
    try:
        state = _get_state()
        data = RecommendRequest.model_validate(request.data)

        recs = get_recommendations(
            history_item_ids=data.item_ids,
            model=state["model"],
            item2id=state["item2id"],
            id2item=state["id2item"],
            max_len=state["model"].max_len,
            device=DEVICE,
            top_k=data.top_k,
        )

        #convert recommendResponse to normal Response
        response = RecommendResponse(user_id=data.user_id, recommendations=[int(r) for r in recs], top_k=data.top_k)
        return Response(response.dict(), status=HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=HTTP_500_INTERNAL_SERVER_ERROR)

#POST /recommend/customer item id's to user 

@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def recommend_by_customer(request):
    try:
        state = _get_state()
        data = CustomerRecommendRequest.model_validate(request.data)
        history = get_customer_last_items(data.customer_id, limit=10)
        is_new_customer = not history

        if is_new_customer:
            history = get_trending_items(limit=10)

        recs = get_recommendations(
        history_item_ids=history,
        model=state["model"],
        item2id=state["item2id"],
        id2item=state["id2item"],
        max_len=state["model"].max_len,
        device=DEVICE,
        top_k=data.top_k,
    )



        response = CustomerRecommendResponse(
            customer_id=data.customer_id,
            history=history,
            recommendations=[int(r) for r in recs],
            top_k=data.top_k,
            is_new_customer=is_new_customer,
        )
        return Response(response.dict(), status=HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=HTTP_500_INTERNAL_SERVER_ERROR)
