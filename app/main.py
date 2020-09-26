from flask import Flask, Response, request
import json
import random
from pymongo import MongoClient
from .QuoteLib import QuoteLib
from .Quote import Quote
from .Engine import Engine
from .EngineDT import EngineDT

app = Flask(__name__)

with app.app_context():
    # Setup
    mongo_uri = ''
    client = MongoClient(mongo_uri)
    db = client.quote_ratings.nlp_quotes
    #nlp_engine = Engine(db, True) # Naive Bayes
    nlp_engine = EngineDT(db, True) # Decision Tree Classifier
    ql = QuoteLib(db)
    
    # Random quote: Return a random quote
    # Returns Quote object
    @app.route('/random')
    def get_quote():
        res = ql.get_random_quote()
        json_res = json.dumps(res.__dict__)
        return Response(json_res, mimetype='application/json')

    # Rate: Rate a quote
    # post body: (author, quote_text, rating, user_id)
    # rating: {-1, 1}
    @app.route('/rate', methods=['POST'])
    def rate_quote():
        req_data = request.get_json()
        quote = Quote('abc', req_data['quote_text'], req_data['author'])
        rating = req_data['rating']
        user_id = request.args.get('user_id')

        ql.rate_quote(quote, rating, user_id)
        return Response()

    # Train: Train the NLP classifier on current data
    # Return the TrainingResult: Accuracy, training data size
    @app.route('/train')
    def train_engine():
        user_id = request.args.get('user_id')
        training_result = nlp_engine.train(user_id)
        json_res = json.dumps(training_result.__dict__)
        return Response(json_res, mimetype='application/json')

    # Predict a quote: Retrieve a random quote and classify it
    @app.route('/predict')
    def predict_quote():
        user_id = request.args.get('user_id')
        random_quote = ql.get_random_quote()
        prediction = nlp_engine.test_quote(random_quote, user_id)
        res = {
            'quote': random_quote.__dict__,
            'prediction': prediction
        }

        return Response(json.dumps(res), mimetype='application/json')
    
    # Funnel: For testing purposes only
    # Funnel a bunch of random quotes with random ratings in
    # Just to build up a larger data set
    @app.route('/funnel')
    def funnel():
        user_id = 'test_user'
        count = 90
        for _ in range(1, count):
            next_quote = ql.get_random_quote()
            num = random.randint(0, 1)
            rating = -1 if num == 0 else 1
            ql.rate_quote(next_quote, rating, user_id)
        return Response(f'{count} records funneled in', mimetype='text/text')
    

