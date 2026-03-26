# German-Frequency-Scanner

This project provides a Python-based tool to scan a given Wikipedia page and all its direct subpages, extract words, and then sort them by frequency. It's specifically designed for the German language, incorporating advanced text processing techniques like lemmatization for verbs and adjectives to ensure accurate frequency counts.

## Key Features & Benefits

*   **Wikipedia Page & Subpage Scanning**: Automatically navigates to a specified Wikipedia page and extracts text content from all direct subpages linked from the main page.
*   **German Text Processing**: Specialized for the German language, leveraging `spaCy` for tokenization and part-of-speech tagging, and custom LSTM models for accurate lemmatization.
*   **Word Frequency Analysis**: Counts and sorts all extracted words by their occurrence frequency, providing insights into the most common vocabulary on the scanned pages.
*   **Advanced Lemmatization**: Utilizes custom Sequence-to-Sequence LSTM models for precise lemmatization of German verbs and adjectives, reducing conjugated/inflected forms to their base (dictionary) form for more accurate frequency statistics.
*   **Training Data Generation**: Includes scripts for web scraping and generating custom training data for the verb and adjective lemmatization models from sources like Wiktionary.
*   **Model Benchmarking**: Provides utility scripts to evaluate the performance and accuracy of the developed lemmatization models.

## Technologies

The project is built primarily with Python and leverages several powerful libraries for web scraping, data manipulation, and natural language processing.

### Languages

*   Python

### Libraries & Tools

*   `torch`: For building and running the LSTM-based lemmatization models.
*   `pandas`: For data manipulation and handling CSV files.
*   `tqdm`: For displaying progress bars during long-running operations.
*   `requests`: For making HTTP requests to fetch web pages.
*   `BeautifulSoup4` (`bs4`): For parsing HTML content and extracting data from web pages.
*   `numpy`: For numerical operations.
*   `fake-useragent`: To generate realistic browser user agents, helping to avoid being blocked by websites during scraping.
*   `spaCy`: A powerful library for advanced natural language processing, used here for German tokenization and part-of-speech tagging (`de_core_news_sm` model).
*   `csv`: For reading and writing CSV files.
*   `pickle`: For serializing and deserializing Python objects, particularly the model encoders.
*   `re`: For regular expression operations.
*   `collections.Counter`: For efficiently counting word frequencies.

## Project Structure

```
├── .gitignore                      # Git ignore file
├── 500_common_german_adjectives.csv  # List of common German adjectives
├── Adj_model_benchmarking.py       # Script to benchmark adjective lemmatization model
├── Gen_Training_data_adjectives-de - Copy.py # Script to generate German adjective training data
├── Gen_Training_data_verbs-de.py   # Script to generate German verb training data
├── German_Verb_Extraction.py       # Utility script for extracting German verbs
├── German_Verb_Training_Data.csv   # Dataset for German verb training
├── Main_de.py                      # Main script for scanning Wikipedia and analyzing frequencies
├── Old_Version/                    # Directory for older versions/scripts
├── Tier_word_frequency             # Placeholder/Output directory for word frequencies
├── Wikipedia Frequency Generator.py # Legacy Wikipedia frequency generator script
├── Verb_Benchmarking.csv           # Benchmarking data for verb model
├── Verb_Transformer-Model-Depreciated.py # Deprecated transformer model for verbs
├── Verb_model_benchmarking.py      # Script to benchmark verb lemmatization model
└── __pycache__/                    # Python bytecode cache directory
    ├── Verb_Neural_Net.cpython-314.pyc
    ├── de_adjective_lstm.cpython-314.pyc
    ├── de_verb_lstm.cpython-314.pyc
    ├── adj_de_training_data.csv    # Adjective training data
    └── de_adjective_lstm.py        # Adjective LSTM lemmatizer implementation
```

## Prerequisites & Dependencies

Before you can run this project, ensure you have Python installed and then install the necessary libraries.

*   **Python 3.x**
*   **Pip** (Python package installer)

## Installation & Setup Instructions

Follow these steps to get the German Frequency Scanner up and running on your local machine.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Ben8547/German-Frequency-Scanner.git
    cd German-Frequency-Scanner
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install torch pandas tqdm requests beautifulsoup4 numpy fake-useragent spacy
    ```

4.  **Download the German spaCy model**:
    This project uses the small German model for `spaCy`.
    ```bash
    python -m spacy download de_core_news_sm
    ```

5.  **Prepare Lemmatization Models**:
    The project expects pre-trained lemmatization models (`de_adj_lemmatizer_model.pt`, `de_verb_lemmatizer_model.pt`) and their respective encoders (`encoder.pkl`) to be present in specific directories (e.g., `./lemmatize_adj_model/`). You will need to either:
    *   **Train your own models**: Use `Gen_Training_data_adjectives-de - Copy.py` and `Gen_Training_data_verbs-de.py` to generate training data, then develop/train the `de_adjective_lstm.py` and `de_verb_lstm.py` models.
    *   **Obtain pre-trained models**: If available, place them in the correct `lemmatize_adj_model` and `lemmatize_verb_model` directories. (These models are not included in the repository by default, implying they need to be generated or downloaded separately.)

## Usage Examples

### Running the German Wikipedia Scanner

To scan a Wikipedia page and its direct subpages, use the `Main_de.py` script.

```bash
python Main_de.py <wikipedia_url>
```

**Example:**
```bash
python Main_de.py "https://de.wikipedia.org/wiki/Deutschland"
```
This will:
1.  Scrape the provided URL and its direct subpages.
2.  Extract text, tokenizing and lemmatizing German words (verbs and adjectives).
3.  Calculate word frequencies.
4.  Output the results (e.g., to a CSV file or console, as implemented in `Main_de.py`).

### Generating Training Data

If you wish to train or re-train the lemmatization models, you can generate fresh training data.

**For Adjectives:**
```bash
python "Gen_Training_data_adjectives-de - Copy.py"
```
This script will scrape Wiktionary or similar sources to create a dataset mapping inflected German adjectives to their base forms.

**For Verbs:**
```bash
python Gen_Training_data_verbs-de.py
```
This script processes raw verb lists (likely generated by `German_Verb_Extraction.py`) to produce training data that maps conjugated German verbs to their infinitive forms.

### Benchmarking Models

To evaluate the performance of your lemmatization models:

**For Adjectives:**
```bash
python Adj_model_benchmarking.py
```

**For Verbs:**
```bash
python Verb_model_benchmarking.py
```
These scripts will load the respective models and run them against a test set, outputting performance metrics.

## Configuration Options

The primary configurable option for the main scanner is the **Wikipedia URL** provided as a command-line argument to `Main_de.py`.

Other "configuration" would involve modifying the Python scripts themselves:

*   **Model Paths**: The paths to the saved LSTM models (`de_adj_lemmatizer_model.pt`, `de_verb_lemmatizer_model.pt`) and their encoders (`encoder.pkl`) are currently hardcoded in scripts like `Adj_model_benchmarking.py` and `Main_de.py`. You would need to edit these files if your models are located elsewhere.
*   **Scraping Parameters**: Scripts like `Gen_Training_data_adjectives-de - Copy.py` might have internal parameters for scraping depth, delay, or specific URLs, which can be adjusted by modifying the code.
*   **Output Format**: The `Main_de.py` script's output format (e.g., CSV columns, sorting order) can be customized by editing the script.

## Contributing Guidelines

We welcome contributions to the German-Frequency-Scanner project! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  **Fork** the repository on GitHub.
2.  **Clone** your forked repository to your local machine.
3.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b bugfix/issue-description`.
4.  **Make your changes**, ensuring that your code adheres to the project's style and conventions.
5.  **Test your changes** thoroughly.
6.  **Commit your changes** with a clear and concise commit message: `git commit -m "feat: Add new feature"`.
7.  **Push your branch** to your forked repository: `git push origin feature/your-feature-name`.
8.  **Open a Pull Request** against the `main` branch of the original repository. Provide a detailed description of your changes.

## License Information

The licensing terms for this project are **not specified**. Please contact the owner for more information regarding usage and distribution rights.
