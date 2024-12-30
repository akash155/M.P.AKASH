A Python-based recommendation system that provides personalized suggestions based on user-item interaction data.

## Features
- User- and item-based collaborative filtering
- Cosine similarity for recommendations
- Evaluation using Mean Squared Error (MSE)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/personalized_recommendation_system.git
   cd personalized_recommendation_system
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\\Scripts\\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Add your dataset as `ratings.csv` in the `data/` folder.
2. Run the script:
   ```bash
   python src/personalized_recommendation_system.py
   ```
3. View recommendations and evaluate the model.

## Project Structure
```
personalized_recommendation_system/
├── data/                     # genome_scores
├── src/                      # Source code
├── notebooks/                # Demonstration notebooks
├── README.md                 # Project overview
├── requirements.txt          # Dependencies
├── .gitignore                # Files to ignore
└── LICENSE                   # License file
```

| User ID | Recommended Items |
|---------|--------------------|
| 101     | [Item 5, Item 8]   |
| 102     | [Item 2, Item 9]   |

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
