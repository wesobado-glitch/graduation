from unittest.mock import patch
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase
from rest_framework import status

class RecommenderTests(APITestCase):
    def setUp(self):
        # Create a test user and token
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key)

    @patch('recommender.views.get_customer_last_items')
    @patch('recommender.views.get_recommendations')
    def test_recommend_by_customer_with_customer_id(self, mock_get_recs, mock_get_history):
        mock_get_history.return_value = [101, 102]
        mock_get_recs.return_value = [201, 202, 203]

        data = {
            "customer_id": 123,
            "top_k": 5
        }
        response = self.client.post('/recommend/customer', data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        res_json = response.json()
        self.assertEqual(res_json['customer_id'], 123)
        self.assertEqual(res_json['history'], [101, 102])
        self.assertEqual(res_json['recommendations'], [201, 202, 203])
        self.assertEqual(res_json['top_k'], 5)
        self.assertEqual(res_json['is_new_customer'], False)

    @patch('recommender.views.get_customer_last_items')
    @patch('recommender.views.get_recommendations')
    def test_recommend_by_customer_with_user_id_alias(self, mock_get_recs, mock_get_history):
        mock_get_history.return_value = [101, 102]
        mock_get_recs.return_value = [201, 202, 203]

        data = {
            "user_id": 123,
            "top_k": 5
        }
        response = self.client.post('/recommend/customer', data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        res_json = response.json()
        self.assertEqual(res_json['customer_id'], 123)
        self.assertEqual(res_json['history'], [101, 102])
        self.assertEqual(res_json['recommendations'], [201, 202, 203])
        self.assertEqual(res_json['top_k'], 5)
        self.assertEqual(res_json['is_new_customer'], False)

    @patch('recommender.views.get_customer_last_items')
    @patch('recommender.views.get_trending_items')
    @patch('recommender.views.get_recommendations')
    def test_recommend_by_customer_new_customer(self, mock_get_recs, mock_get_trending, mock_get_history):
        mock_get_history.return_value = []
        mock_get_trending.return_value = [501, 502]
        mock_get_recs.return_value = [301, 302]

        data = {
            "user_id": 456,
            "top_k": 2
        }
        response = self.client.post('/recommend/customer', data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        res_json = response.json()
        self.assertEqual(res_json['customer_id'], 456)
        self.assertEqual(res_json['history'], [501, 502])
        self.assertEqual(res_json['recommendations'], [301, 302])
        self.assertEqual(res_json['top_k'], 2)
        self.assertEqual(res_json['is_new_customer'], True)
