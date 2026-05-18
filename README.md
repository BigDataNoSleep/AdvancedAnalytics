# Advanced Analytics in Business

This repository contains the codebase and project files for the "Advanced Analytics in Business" course. The project is divided into four main tasks, focusing on building end-to-end data pipelines, predictive models, and recommender systems.

## Task 1: Customer Lifetime Value (CLV) Prediction
Predicting customer lifetime value and revenue using an ensemble of LightGBM, Hurdle, and Tweedie models. This task focuses on weight optimization and behavioral post-processing to handle zero-inflation bias and achieve the lowest possible Mean Absolute Error (MAE).

## Task 2: Exploratory Data Analysis & Data Leakage
Exploratory Data Analysis (EDA) on transactional data, focusing on mitigating data leakage risks between training and test sets and preparing robust feature engineering pipelines to protect the integrity of the predictive models.

## Task 3: Steam Game Recommendation Engine
A high-performance Retrieval-Augmented Generation (RAG) system for Steam game recommendations. It implements Hybrid Search (BM25 + Vector), self-querying metadata filtering, Cross-Encoder re-ranking, and utilizes local ChromaDB for embeddings.

## Task 4: Storytelling Analysis (Epstein Files)
An analytical deep-dive into the Epstein files to uncover facts and present them through a compelling storytelling narrative. *(Note: This task focuses on qualitative analysis and storytelling, and therefore does not contain code in this repository).*

## Repository Structure
- `task1/`: Code, models, and submission files for CLV prediction.
- `task2/`: EDA and feature engineering scripts.
- `task3/`: RAG recommendation engine implementation and documentation.
- `report/`: Documentation, reports, and guidelines for the project.