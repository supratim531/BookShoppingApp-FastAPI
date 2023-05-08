import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chatbot.chat import answer_of_chatbot as answer


app = FastAPI()
books = pd.read_pickle("books.pkl")
final_df = pd.read_pickle("final_df.pkl")
similarity_score = pd.read_pickle("similarity_score.pkl")
popular_50_unique_books_extracted_info = pd.read_pickle(
    "popular_50_unique_books_extracted_info.pkl")


origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def recommend_books_info(book_name):
    # index fetch
    index = np.where(final_df.index == book_name)[0][0]
    similer_items = sorted(
        list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]

    data = []
    for i in similer_items:
        item = []
        temp_df = books[books["Book-Title"] == final_df.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates(
            "Book-Title")["Book-Title"].values))
        item.extend(list(temp_df.drop_duplicates(
            "Book-Title")["Book-Author"].values))
        item.extend(list(temp_df.drop_duplicates(
            "Book-Title")["Image-URL-L"].values))
        item.extend(list(temp_df.drop_duplicates(
            "Book-Title")["Publisher"].values))
        data.append(item)

    return data


@app.get('/')
def index():
    return {
        "message": "FastAPI server is running"
    }


@app.get("/api/top-50-books")
def top_50_books():
    data = []
    bookNames = list(
        popular_50_unique_books_extracted_info["Book-Title"].values)
    authors = list(
        popular_50_unique_books_extracted_info["Book-Author"].values)
    bookImages = list(
        popular_50_unique_books_extracted_info["Image-URL-L"].values)
    ratings = list(
        popular_50_unique_books_extracted_info["Average-Rating"].values)

    for i in range(0, 50):
        data.append({
            "bookName": bookNames[i],
            "author": authors[i],
            "bookImage": bookImages[i],
            "rating": ratings[i]
        })

    return data


@app.get("/api/recommend-books-info/{bookName}")
def recommended_book(bookName: str):
    try:
        data = []
        result = recommend_books_info(bookName)

        for i in result:
            data.append({
                "bookName": i[0],
                "author": i[1],
                "bookImage": i[2],
                "publisher": i[3]
            })
    except Exception as e:
        print(e)
        return {
            "data": None
        }

    return data


@app.get("/api/assistant-response")
def assistant_response(query: str):
    result = None

    try:
        result = answer(query)
        print(result)
    except Exception as e:
        print(e)

    return {
        "data": result
    }
