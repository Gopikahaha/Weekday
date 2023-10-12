import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import re


def extract_and_save_random_rows(input_file, output_file, num_random_rows=20, min_rows_per_category=2):
    try:
        # Read the original CSV file
        original_df = pd.read_csv(input_file)

        # Initialize a dictionary to keep track of how many rows are selected for each category
        category_counts = {category: 0 for category in categories_keywords}

        # List to store the selected rows
        selected_rows = []

        # Iterate through the rows
        for _, row in original_df.iterrows():
            # Check for missing values in the 'Description' column
            description = row['Description']
            if not pd.isna(description) and len(str(description).strip()) > 0:
                # Get the categories of the row (split by comma)
                categories = row['Category']
                if pd.isna(categories):
                    categories = ""
                categories = [cat.strip() for cat in str(categories).split(',')]

                for category in categories:
                    if category in categories_keywords:
                        # Check if the category has not exceeded the required minimum rows
                        if category_counts[category] < min_rows_per_category:
                            selected_rows.append(row)
                            category_counts[category] += 1
                        break  # Stop checking other categories for this row

        # Create a new DataFrame with the selected rows
        random_df = pd.DataFrame(selected_rows)

        # Save the random data to a new CSV file
        random_df.to_csv(output_file, index=False)

        print(
            f'{len(selected_rows)} random rows with at least {min_rows_per_category} rows per category saved to {output_file}')
    except Exception as e:
        print(f"An error occurred: {e}")


# Define your categories and their corresponding keywords
categories_keywords = {
    "Product based": ["product", "software", "scope", "products", "demonstration", "built", "standalone", "SaaS",
                      "developed"],
    "Service based": ["service", "servicing", "consulting", "support", "solutions", "enables", "enable", "promote",
                      "qualified", "helps", "bring", "SaaS", "build", "conduct", "agents", "offers", "mentorship"],
    "Fintech": ["fintech", "finance", "payments", "banking", "financial"],
    "Edtech": ["edtech", "ed-tech", "education", "educational", "e-learning", "online courses", "Academy"],
    "Big Bank": ["investment", "corporate banking"],
    "AI": ["artificial intelligence", "machine learning", "AI", "ML"],
    "Marketplace": ["delivery partners", "partnered", "marketplace"],
    "Foreign": []
}


def categorize(row):
    categories_set = set()  # Use a set to ensure unique categories
    description = str(row["Description"]).lower()

    for category, keywords in categories_keywords.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, description):
                categories_set.add(category)  # Add matched category to the set

    # Check the Country column and add 'Foreign company' to the set if it's not 'IN' (India)
    if pd.notna(row["Country"]) and row["Country"] != "IN":
        categories_set.add("Foreign company")

    return ",".join(categories_set)  # Convert the set to a comma-separated string


def display_and_confirm_categories(data_file):
    try:
        random_df = pd.read_csv(data_file)

        for index, row in random_df.iterrows():
            print(f"Company Name: {row['Company Name']}")
            print(f"Description:\n{row['Description']}")
            current_categories = [category.strip() for category in row['Category'].split(',')]
            print(f"Current Categories: {', '.join(current_categories)}")

            user_input = input("Enter the updated categories (e.g., '1,2,3') or '0' to keep: ")

            if user_input == '0':
                continue
            else:
                new_category_indices = [int(cat) - 1 for cat in user_input.split(',') if cat.isdigit()]
                new_categories = [list(categories_keywords.keys())[idx] for idx in new_category_indices]
                random_df.at[index, 'Category'] = ', '.join(new_categories)

        # Save the updated categories
        random_df.to_csv(data_file, index=False)
        print("Categories updated and saved to the file.")

    except Exception as e:
        print(f"An error occurred: {e}")


# Function to add a category if it's not already present
def add_category_if_not_present(categories, category_to_add):
    if category_to_add not in categories:
        categories.append(category_to_add)
    return categories


# Function to add "AI" category if "AI" or "ML" is found in any column (case-sensitive)
def add_ai_category(row, categories):
    if ('AI' in row or 'ML' in row) and 'AI' not in categories:
        categories = add_category_if_not_present(categories, 'AI')
    return categories


# Function to add "Fintech" category if "fintech" is found in any column (case-insensitive)
def add_fintech_category(row, categories):
    if re.search(r'\bfintech\b', row, re.IGNORECASE) and 'Fintech' not in categories:
        categories = add_category_if_not_present(categories, 'Fintech')
    return categories


# Function to add "Edtech" category if "edtech" is found in any column (case-insensitive)
def add_edtech_category(row, categories):
    if re.search(r'\bedtech\b', row, re.IGNORECASE) and 'Edtech' not in categories:
        categories = add_category_if_not_present(categories, 'Edtech')
    return categories


def remove_common_and_save_uncommon(file1, file2, output_file):
    try:
        # Read the two CSV files into DataFrames
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Exclude the 'Category' column from both DataFrames for comparison
        common_columns = [col for col in df1.columns if col != 'Category']
        df1 = df1[common_columns]
        df2 = df2[common_columns]

        # Merge the DataFrames with an indicator to track common rows
        merged = pd.merge(df1, df2, how='outer', indicator=True)

        # Filter rows to keep only those not in both dataframes
        uncommon_data = merged[merged['_merge'] == 'left_only']

        # Drop the indicator column and save the uncommon rows to a new CSV file
        uncommon_data = uncommon_data.drop(columns=['_merge'])
        uncommon_data.to_csv(output_file, index=False)
        print(f"Uncommon rows saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


def append_csv_files(file1, file2, output_file):
    try:
        # Read the first CSV file into a DataFrame
        df1 = pd.read_csv(file1)

        # Read the second CSV file, without skipping the first row
        df2 = pd.read_csv(file2)

        # Check if the columns in both DataFrames match
        if list(df1.columns) != list(df2.columns):
            raise ValueError("Columns in the two CSV files do not match.")

        # Append the data from the second DataFrame to the first DataFrame
        appended_data = pd.concat([df1, df2.iloc[0:]], ignore_index=True)

        # Save the appended data to a new CSV file
        appended_data.to_csv(output_file, index=False)
        print(f'Appended data saved to {output_file}')

    except Exception as e:
        print(f"An error occurred: {e}")


def train():
    # Load the CSV files
    training_data = pd.read_csv(labelled_companies_file)  # First CSV file with categorized companies
    unlabeled_data = pd.read_csv(unlabelled_companies_file)  # Second CSV file with uncategorized companies

    # Fill missing or NaN values in all columns with empty strings
    training_data = training_data.fillna('')
    unlabeled_data = unlabeled_data.fillna('')

    # Combine 'Enrich Company,' 'Industry,' 'Country,' and 'Description' into a single text column
    training_data['Combined_Text'] = training_data['Company Name'] + ' ' + training_data['Industry'] + ' ' + \
                                     training_data['Description']
    unlabeled_data['Combined_Text'] = unlabeled_data['Company Name'] + ' ' + unlabeled_data['Industry'] + ' ' + \
                                      unlabeled_data['Description']

    # Convert the 'Category' column to lists, handling non-string values
    y_train = [str(categories).split(',') if isinstance(categories, str) else [] for categories in
               training_data['Category']]

    # Convert the multilabel categories into binary labels
    mlb = MultiLabelBinarizer()
    y_train_binarized = mlb.fit_transform(y_train)

    # Prepare the training data
    X_train = training_data['Combined_Text']

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features as needed

    # Fit and transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Train a Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_tfidf, y_train_binarized)

    # Transform the unlabeled data
    X_unlabeled_tfidf = tfidf_vectorizer.transform(unlabeled_data['Combined_Text'])

    # Predict categories for the unlabeled data
    y_pred_binarized = classifier.predict(X_unlabeled_tfidf)

    # Convert the binary labels back to multilabel categories
    y_pred_categories = mlb.inverse_transform(y_pred_binarized)

    # Function to add a category if it's not already present
    # Check if "IN" is in the 'Country' column, if not, add "Foreign company" to categories
    unlabeled_data['Category'] = [
        ','.join(add_ai_category(row, add_fintech_category(row, add_edtech_category(row, list(categories))))) + (
            ',Foreign' if 'IN' not in country and 'Foreign' not in categories else '') for categories, row, country in
        zip(y_pred_categories, unlabeled_data['Combined_Text'], unlabeled_data['Country'])]

    unlabeled_data = unlabeled_data.drop(columns=['Combined_Text'])

    # Save the results to a new CSV file
    unlabeled_data.to_csv(keyword_categorised_companies_file, index=False)


def create_final_file():
    # Load the "Weekday_Companies" and "CategorisedCompanies" CSV files
    weekday_companies = pd.read_csv("Weekday_Companies.csv")
    categorised_companies = pd.read_csv("CategorisedCompanies.csv")

    # Define the keywords and columns to map
    keywords = ["Service based", "Product based", "Fintech", "Big Bank", "Foreign", "Edtech", "A.I.", "Marketplace"]
    columns_to_map = ["Services company", "Product based", "Fintech", "Big Bank", "Foreign companies", "Edtech", "A.I.",
                      "Marketplace"]

    # Iterate through rows in "CategorisedCompanies" to update checkboxes in "Weekday_Companies"
    for index, row in categorised_companies.iterrows():
        # Get the values from the "Weekday_Companies" and "CategorisedCompanies" data
        category_column = row['Category']
        first_column = row['Company Name']

        # Find the corresponding row in "Weekday_Companies" based on the matching criteria
        match = weekday_companies[
            (weekday_companies['Company Name'] == first_column)
        ]

        # If a match is found, update the checkbox columns based on the presence of keywords
        if not match.empty:
            for keyword, column_name in zip(keywords, columns_to_map):
                # Check if the "Category" value is not NaN
                if pd.notna(category_column):
                    weekday_companies.loc[match.index, column_name] = keyword in category_column

    # Save the updated "Weekday_Companies" to a new CSV file
    weekday_companies.to_csv("Updated_Weekday_Companies.csv", index=False)


# Usage:
companies_file = 'Companies.csv'
keyword_labelled_companies_file = 'KeywordLabelledCompanies.csv'
keyword_categorised_companies_file = "KeywordCategorisedCompanies.csv"
unlabelled_companies_file = "UnlabelledCompanies.csv"
labelled_companies_file = "LabelledCompanies.csv"
categorised_companies_file = "CategorisedCompanies.csv"

df = pd.read_csv(companies_file)
df["Category"] = df.apply(categorize, axis=1)
df.to_csv(keyword_categorised_companies_file, index=False)
num_rows = 20
min_rows_per_category = 2

for i in range(10):
    extract_and_save_random_rows(keyword_categorised_companies_file, keyword_labelled_companies_file)
    display_and_confirm_categories(keyword_labelled_companies_file)
    append_csv_files(labelled_companies_file, keyword_labelled_companies_file, labelled_companies_file)
    remove_common_and_save_uncommon(companies_file, labelled_companies_file, unlabelled_companies_file)
    train()

append_csv_files(labelled_companies_file, keyword_categorised_companies_file, categorised_companies_file)
create_final_file()