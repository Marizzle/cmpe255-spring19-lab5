from collections import Counter
from linear_algebra import distance
from stats import mean
import math, random
import matplotlib.pyplot as plt


def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest

def knn_classify(k, labeled_points, new_point):
    from sklearn.neighbors import KNeighborsClassifier
    X = [labeled_point[0] for labeled_point in labeled_points]
    y = [labeled_point[1] for labeled_point in labeled_points]

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)

    return neigh.predict([new_point])


def predict_preferred_language_by_city(k_values, cities):
    """
    TODO
    predicts a preferred programming language for each city using above knn_classify() and
    counts if predicted language matches the actual language.
    Finally, print number of correct for each k value using this:
    print(k, "neighbor[s]:", num_correct, "correct out of", len(cities))
    """
    from copy import deepcopy
    for k_value in k_values:
        num_correct = 0
        for city in cities:

            citieswithoutcurrentcity = deepcopy(cities)
            citieswithoutcurrentcity.remove(city)

            city_lat_long = city[0]
            prediction = knn_classify(k_value, citieswithoutcurrentcity, city_lat_long)

            # if prediction matched label, increment correct
            if prediction == city[1]:
                num_correct += 1

        print("{} neighbor[s]: {} correct out of {}".format(k_value, num_correct, len(cities)))

if __name__ == "__main__":
    k_values = [1, 3, 5, 7]
    # TODO
    # Import cities from data.py and pass it into predict_preferred_language_by_city(x, y). (BOOOOOM)
    from data import cities
    predict_preferred_language_by_city(k_values, cities)
