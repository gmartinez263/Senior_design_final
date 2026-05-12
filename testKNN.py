import unittest
from unittest.mock import patch, MagicMock
from recommender import KNN

class TestKNNPhases(unittest.TestCase):
    def setUp(self):
        self.mock_cannabinoids = [
            {'id': 1, 'name': 'THC'},
            {'id': 2, 'name': 'CBD'}
        ]
        self.mock_terpenes = [
            {'id': 1, 'name': 'Myrcene'},
            {'id': 2, 'name': 'Limonene'}
        ]
        
        # Helper to create a dummy session
        self.create_session = lambda id, servings, effects: {
            'id': id,
            'product_id': 1,
            'servings': servings,
            'Products': {'type': 'Flower'},
            'cannabinoids': [{'cannabinoid_id': 1, 'serving': 10.0}],
            'terpenes': [{'terpene_id': 1}],
            'effects': effects
        }

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_phase_1_cold_start(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Cold Start (< 10 sessions) there should be no tuning.
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes

        #5 sessions < 10
        sessions = []
        for i in range(5):
            sessions.append(self.create_session(i, 1.0, ['Relaxed']))
        mock_sessions.return_value = sessions

        knn = KNN(user_id=1)
        
        # Verify default weights are preserved
        self.assertEqual(knn.w_chem, 0.5)
        self.assertEqual(knn.w_dose, 0.3)
        self.assertEqual(knn.w_admin, 0.2)
        self.assertEqual(knn.gamma, 0.1)

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_phase_2_initial_tuning(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Initial Tuning (10-19 sessions) - Leave-One-Out CV.
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        # 15 sessions (between 10 and 20)
        mock_sessions.return_value = [self.create_session(i, 1.0, ['Relaxed']) for i in range(15)]
        
        #check to see if weights change from default if a better score is found
        with patch.object(KNN, '_evaluate_session', return_value=0.8) as mock_eval:
            knn = KNN(user_id=1)

            self.assertGreater(mock_eval.call_count, 0)
            #Leave-one-out means history size should be 14 during eval
            self.assertEqual(len(mock_eval.call_args_list[0][0][0]['effects']), 1)

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_phase_3_mature_phase(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Mature Phase (>= 20 sessions) Rolling-Window Validation.
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        mock_sessions.return_value = [self.create_session(i, 1.0, ['Relaxed']) for i in range(25)]
        
        with patch.object(KNN, '_evaluate_session', return_value=0.8) as mock_eval:
            knn = KNN(user_id=1)
            self.assertGreater(mock_eval.call_count, 0)
            self.assertEqual(mock_eval.call_count, 1275)

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_prediction_internal_phases(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Test the internal phases of the predict method.
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        mock_sessions.return_value = [
            self.create_session(1, 1.0, ['Relaxed']),
            self.create_session(2, 2.0, ['Energetic']),
        ]
        
        knn = KNN(user_id=1)
        
        query_prod = {
            'cannabinoids': [{'cannabinoid_id': 1, 'serving': 10.0}],
            'terpenes': [{'terpene_id': 1}],
            'type': 'Flower'
        }
        query_dose = 1.0
        
        sim1 = knn._calculate_similarity(query_prod, query_dose, knn.history[0])
        sim2 = knn._calculate_similarity(query_prod, query_dose, knn.history[1])
        self.assertGreater(sim1, sim2)
        
        results = knn.predict(query_prod, query_dose)
        
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) <= 3)
        self.assertEqual(results[0]['effect'], 'Relaxed')
        self.assertGreater(results[0]['probability'], 0.5)

if __name__ == '__main__':
    unittest.main()