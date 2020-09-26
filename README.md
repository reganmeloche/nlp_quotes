## NLP Quotes

This is a project that uses Natural Language Processing techniques to predict quotes that a given user will like.


### Endpoints
- `GET /random`: Request a random quote (powered by wikiquote)

- `POST /rate`: rate the quote, using the following body (rating = 1 for like, -1 for dislike):
```
{
    "quote_text": "{{quote_text}}",
    "author": "{{author}}",
    "user_id": "{{user_id}}",
    "rating": -1 
}
```

- `/train`: Train a model using the list of quotes liked and disliked by a user

- `/predict`: Get a prediction on another randomly generated quote

*** At this point, the predict, train, and rate endpoints must be called with a user_id query parameter


### Todo
- Pull out the static feature extractors and add tests
- Custom feature engineering (author, themes, etc)
- Add explicit types on the args and function results
- Check to see if a quote is already stored for a user
- Front-end: create a proper form to display the quote, and user can click to save
- Improve parsing of the quote text
- Add in more user-based functionality... login, sessions, etc.
- Add in config variables


### Sources
- Modeled off of examples in this book: https://www.manning.com/books/getting-started-with-natural-language-processing
- For extracting random quotes from wikiquote: https://codepen.io/Eatcake/pen/ggxYeN?editors=1010