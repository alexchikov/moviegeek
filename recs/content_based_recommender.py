from decimal import Decimal
from django.db.models import Q

from analytics.models import Rating
from recommender.models import MovieDescriptions, LdaSimilarity
from moviegeeks.models import Movie
from recs.base_recommender import base_recommender

lda_path = './lda/'


class ContentBasedRecs(base_recommender):

    def __init__(self, min_sim=0.1):
        self.min_sim = min_sim
        self.max_candidates = 100

    def recommend_items(self, user_id, num=6):
        active_user_items = Rating.objects.filter(user_id=user_id).order_by('-rating')[:100]
        return self.recommend_items_by_ratings(user_id, active_user_items.values(), num)

    @staticmethod
    def seeded_rec(content_ids, take=6):
        data = LdaSimilarity.objects.filter(source__in=content_ids) \
                   .order_by('-similarity') \
                   .values('target', 'similarity')[:take]
        return list(data)

    def recommend_items_by_ratings(self, user_id, active_user_items, num=6):
        if len(active_user_items) == 0:
            return {}

        movie_ids = {movie['movie_id']: movie['rating'] for movie in active_user_items}
        user_mean = sum(movie_ids.values()) / len(movie_ids)

        sims = LdaSimilarity.objects.filter(
            Q(source__in=movie_ids.keys()) &
            ~Q(target__in=movie_ids.keys()) &
            Q(similarity__gt=self.min_sim)
        ).order_by('-similarity')[10:self.max_candidates]

        valid_movie_ids = set(Movie.objects.values_list('movie_id', flat=True))

        recs = dict()
        targets = set(s.target for s in sims if s.target and s.target in valid_movie_ids)

        for target in targets:
            pre = 0
            sim_sum = 0
            rated_items = [i for i in sims if i.target == target]

            if rated_items:
                for sim_item in rated_items:
                    r = Decimal(movie_ids[sim_item.source] - user_mean)
                    pre += sim_item.similarity * r
                    sim_sum += sim_item.similarity

                if sim_sum > 0:
                    recs[target] = {
                        'prediction': Decimal(user_mean) + pre / sim_sum,
                        'sim_items': [r.source for r in rated_items]
                    }

        return sorted(recs.items(), key=lambda item: -float(item[1]['prediction']))[:num]

    def predict_score(self, user_id, item_id):
        user_items = (Rating.objects.filter(user_id=user_id)
                      .exclude(movie_id=item_id)
                      .order_by('-rating').values()[:100])

        movie_ids = {movie['movie_id']: movie['rating'] for movie in user_items}
        user_mean = sum(movie_ids.values()) / len(movie_ids)

        sims = LdaSimilarity.objects.filter(
            Q(source__in=movie_ids.keys()) &
            Q(target=item_id) &
            Q(similarity__gt=self.min_sim)
        ).order_by('-similarity')

        pre = 0
        sim_sum = 0
        prediction = Decimal(0.0)

        if sims.exists():
            for sim_item in sims:
                r = Decimal(movie_ids[sim_item.source] - user_mean)
                pre += sim_item.similarity * r
                sim_sum += sim_item.similarity

            prediction = Decimal(user_mean) + pre / sim_sum

        return prediction
