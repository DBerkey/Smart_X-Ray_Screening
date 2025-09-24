"""
Data Analysis Script - Minimized with comments
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


source_path = 'path/to/Data_Entry_2017_v2020.csv'  # Replace with the path to your CSV file
store_path = 'path/to/store/report'  # Replace with the path to store the report


def age_and_gender_distribution(df_input):
    """
    Plot age vs gender distribution as line chart

    Mathematical Operation:
    - For each age a ∈ {min(age), ..., max(age)}:
      - Count_M(a) = |{patient i | age_i = a AND sex_i = 'M'}|
      - Count_F(a) = |{patient i | age_i = a AND sex_i = 'F'}|
    - Plot: (age, Count_M) and (age, Count_F) as line graphs

    Set Theory: Partition patients P by age and gender: P = ⋃(a,s) P_{a,s}
    where P_{a,s} = {p ∈ P | age(p) = a ∧ sex(p) = s}
    """
    # Group by age and gender, create pivot table
    age_gender = df_input.groupby(
        ['Patient Age', 'Patient Sex']).size().unstack(fill_value=0)
    ages = age_gender.index.tolist()
    males = age_gender.get('M', pd.Series([0]*len(ages), index=ages)).tolist()
    females = age_gender.get('F', pd.Series(
        [0]*len(ages), index=ages)).tolist()

    # Plot lines for each gender
    plt.figure(figsize=(12, 7))
    plt.plot(ages, males, color='blue', label='Male', linewidth=2)
    plt.plot(ages, females, color='red', label='Female', linewidth=2)
    plt.xlabel('Age', fontsize=12), plt.ylabel('Count', fontsize=12)
    plt.title('Age and Gender Distribution', fontsize=14)
    plt.legend(fontsize=11), plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(store_path + 'age_and_gender_distribution.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def image_variation(df_input):
    """
    Analyze image characteristics: view positions, sizes, pixel spacing

    Mathematical Operations:
    1. View Position Distribution: 
       - P(view_i) = |{images with view_i}| / |total_images|
       - Pie chart shows: ∑P(view_i) = 1

    2. Image Size Distribution:
       - Size_i = width_i × height_i (concatenated as string)
       - Count(size_j) = |{images with size_j}|
       - Bar chart: frequency distribution over size space

    3. Pixel Spacing Distribution:
       - Spacing_i = (x_spacing_i, y_spacing_i)
       - Count(spacing_j) = |{images with spacing_j}|
       - Bar chart: frequency distribution over spacing space
    """
    # View positions pie chart
    plt.figure(figsize=(8, 8))
    view_counts = df_input['View Position'].value_counts()
    view_counts.plot(kind='pie', colors=[
                     '#1f77b4', '#ff7f0e'], autopct='%1.1f%%', startangle=90)
    plt.title('Image Variation Distribution', fontsize=14), plt.ylabel('')
    plt.tight_layout()
    plt.savefig(store_path + 'view_position_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Image sizes bar chart (top 15 + others for readability)
    df_input['ImageSize'] = df_input['OriginalImage[Width'].astype(
        str) + 'x' + df_input['Height]'].astype(str)
    size_counts = df_input['ImageSize'].value_counts()
    top_sizes = size_counts.head(15)  # Reduced to 15 for better readability
    others = size_counts.iloc[15:].sum()

    labels = top_sizes.index.tolist()
    counts = top_sizes.values.tolist()
    if others > 0:
        labels.append('Others'), counts.append(others)

    plt.figure(figsize=(16, 8))  # Wider figure
    plt.bar(labels, counts, color='#2ca02c')
    # Right-align rotated text
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title('Top 15 Image Sizes', fontsize=14)
    plt.xlabel('Image Size', fontsize=12), plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(store_path + 'image_sizes_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Pixel spacing distribution
    plt.figure(figsize=(14, 7))
    df_input['PixelSpacing'] = df_input['OriginalImagePixelSpacing[x'].astype(
        str) + ',' + df_input['y]'].astype(str)
    df_input['PixelSpacing'].value_counts().plot(kind='bar', color='#9467bd')
    plt.title('Pixel Spacing Distribution', fontsize=14)
    plt.xlabel('Pixel Spacing', fontsize=12), plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(store_path + 'pixel_spacing_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def age_vs_amount_of_diseases(df_input):
    """
    Plot average number of diseases per age group

    Mathematical Operation:
    - For patient i: diseases_i = |split(Finding_Labels_i, '|')| if not null, else 0
    - For age group a: Avg_Diseases(a) = (∑_{i: age_i=a} diseases_i) / |{i: age_i=a}|
    - Plot: (age, Avg_Diseases(age)) as line graph

    Formula: μ_a = (1/n_a) * ∑_{i∈A_a} d_i
    where A_a = {patients with age a}, n_a = |A_a|, d_i = number of diseases for patient i
    """
    # Count diseases per patient (split by '|')
    df_input['Num_Diseases'] = df_input['Finding Labels'].apply(
        lambda x: len(x.split('|')) if pd.notnull(x) else 0)
    # Calculate mean diseases per age
    age_disease = df_input.groupby('Patient Age')['Num_Diseases'].mean()

    plt.figure(figsize=(12, 7))
    plt.plot(age_disease.index, age_disease.values, marker='o',
             linewidth=2, markersize=4, color='#d62728')
    plt.xlabel('Age', fontsize=12), plt.ylabel('Avg Diseases', fontsize=12)
    plt.title('Age vs Disease Count', fontsize=14)
    plt.grid(True, alpha=0.3), plt.tight_layout()
    plt.savefig(store_path + 'age_vs_disease_count.png', dpi=300, bbox_inches='tight')
    plt.show()


def disease_co_occurrence(df_input):
    """
    Analyze which diseases occur together in same patients

    Mathematical Operation:
    - Let D = {all unique diseases} be the disease universe
    - For diseases d1, d2 ∈ D: Co_Matrix[d1][d2] = |{patients with both d1 and d2}|
    - Matrix M where M_{ij} = |{p ∈ P | d_i ∈ diseases(p) ∧ d_j ∈ diseases(p) ∧ d_i ≠ d_j}|
    - Stacked bar chart: for each primary disease d_i, show distribution of co-occurring diseases

    Set Theory: For patient p, diseases(p) = split(Finding_Labels_p, '|')
    Co-occurrence relation: R = {(d1,d2) | ∃p: d1,d2 ∈ diseases(p) ∧ d1≠d2}
    """
    # Get all unique diseases
    all_diseases = set()
    for labels in df_input['Finding Labels'].dropna():
        diseases = labels.split('|')
        all_diseases.update(diseases)
    all_diseases = sorted(list(all_diseases))
    print(f"Unique diseases found: {len(all_diseases)}")


    # Create co-occurrence matrix
    co_matrix = {d1: {d2: 0 for d2 in all_diseases} for d1 in all_diseases}

    # Count disease pairs in same patient
    for labels in df_input['Finding Labels'].dropna():
        diseases = labels.split('|')
        if len(diseases) < 2:
            co_matrix[diseases[0]][diseases[0]] += 1 #count how often a disease occurs alone
            continue
        for d1 in diseases:
            for d2 in diseases:
                if d1 != d2:
                    co_matrix[d1][d2] += 1
    
    #normalize
    for d1 in all_diseases:
        total = sum(co_matrix[d1].values())
        if total > 0:
            for d2 in all_diseases:
                co_matrix[d1][d2] = co_matrix[d1][d2] / total * 100

    #make heatmap and put the percentage values in the boxes
    co_df = pd.DataFrame(co_matrix).T
    plt.figure(figsize=(12, 6))
    sns.heatmap(co_df, annot=True, fmt=".1f", cmap='Blues', cbar_kws={'label': 'Percentage'})
    plt.title('Disease Co-Occurrence Heatmap (%)', fontsize=16)
    plt.xlabel('Co-Occurring Disease', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Primary Disease', fontsize=12)
    plt.tight_layout()
    plt.savefig(store_path + 'disease_co_occurrence_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def disease_view_position_analysis(df_input):
    """
    Analyze correspondence between diseases and image view positions - minimal version

    Mathematical Operations:
    1. Data Preparation:
       - For each record i: extract all diseases d ∈ split(Finding_Labels_i, '|') 
       - If Finding_Labels_i is null: disease = 'No Finding'
       - Create pairs: (disease_j, view_position_i)

    2. Cross-tabulation Matrix:
       - M[d][v] = |{records where disease=d AND view_position=v}|
       - Normalization: M_norm[d][v] = M[d][v] / ∑_d M[d][v] * 100
       - This gives column-wise percentages: for each view position, 
         what percentage each disease represents

    3. Mathematical Interpretation:
       - P(disease=d | view=v) = M[d][v] / ∑_d M[d][v]
       - Heatmap shows conditional probability distribution

    Set Theory: 
    - Disease-View pairs: R = {(d,v) | ∃record: d ∈ diseases(record) ∧ v = view(record)}
    - Partition by view: R_v = {(d,v) ∈ R | view = v}
    """
    # Prepare data
    diseases = []
    view_positions = []

    for _, row in df_input.iterrows():
        view_pos = row['View Position']
        if pd.notnull(row['Finding Labels']):
            for disease in row['Finding Labels'].split('|'):
                diseases.append(disease.strip())
                view_positions.append(view_pos)
        else:
            diseases.append('No Finding')
            view_positions.append(view_pos)

    # Create DataFrame and crosstab
    data = pd.DataFrame({'Disease': diseases, 'View_Position': view_positions})
    crosstab = pd.crosstab(
        data['Disease'], data['View_Position'], normalize='columns') * 100

    # Visualization using matplotlib only
    plt.figure(figsize=(10, 8))

    # Create heatmap manually
    im = plt.imshow(crosstab.values, cmap='YlOrRd', aspect='auto')

    # Set ticks and labels
    plt.xticks(range(len(crosstab.columns)), crosstab.columns)
    plt.yticks(range(len(crosstab.index)), crosstab.index)

    # Add text annotations
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            plt.text(j, i, f'{crosstab.iloc[i, j]:.1f}',
                     ha='center', va='center', fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Percentage')

    plt.title('Disease Distribution by View Position (%)')
    plt.xlabel('View Position'), plt.ylabel('Disease')
    plt.tight_layout()
    plt.savefig(store_path + 'disease_view_position_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def disease_age_and_sex_analysis(df_input):
    df = df_input
    new_df = pd.DataFrame()
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(f"Processing row {index}...")
        age = row['Patient Age'] - row['Patient Age'] % 5  # Group ages in 5-year bins
        findings = row['Finding Labels'].strip().split('|')
        for finding in findings:
            new_row = row.copy()
            new_row['Finding Labels'] = finding
            new_row['Patient Age'] = age
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    #group by age and disease
    age_disease = new_df.groupby(['Patient Age', 'Finding Labels']).size().reset_index(name='count')
    #plot heatmap for age vs disease
    age_disease_pivot = age_disease.pivot(index='Finding Labels', columns='Patient Age', values='count').fillna(0)
    print(age_disease_pivot.T)
    for age in age_disease_pivot.columns:
        new_name = f"{age}, {age_disease_pivot[age].sum()} values"
        age_disease_pivot[age] = age_disease_pivot[age] / age_disease_pivot[age].sum() * 100
        age_disease_pivot.rename(columns={age: new_name}, inplace=True)

    plt.figure(figsize=(12, 12))
    sns.heatmap(age_disease_pivot.T, cmap='viridis', annot=False, cbar_kws={'label': 'Distribution (%)'})
    plt.title('Disease Distribution by Age')
    plt.xlabel('Disease')
    plt.ylabel('Age Group')
    plt.tight_layout()
    plt.savefig(store_path + 'disease_age_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


    #plot sex vs disease
    sex_disease = new_df.groupby(['Patient Sex', 'Finding Labels']).size().reset_index(name='count')
    sex_disease_pivot = sex_disease.pivot(index='Patient Sex', columns='Finding Labels', values='count').fillna(0)
    print(sex_disease_pivot)
    for sex in sex_disease_pivot.index:
        print(sex)
        new_name = f"{sex}, {sex_disease_pivot.loc[sex].sum()} values"
        sex_disease_pivot.loc[sex] = sex_disease_pivot.loc[sex] / sex_disease_pivot.loc[sex].sum() * 100
        sex_disease_pivot.rename(index={sex: new_name}, inplace=True)

    plt.figure(figsize=(12, 6))
    sns.heatmap(sex_disease_pivot, cmap='magma', annot=False, cbar_kws={'label': 'Distribution (%)'})
    plt.title('Disease Distribution by sex')
    plt.xlabel('Disease')
    plt.ylabel('Sex')
    plt.tight_layout()
    plt.savefig(store_path + 'disease_sex_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Execute analysis
# Load dataset
df = pd.read_csv(source_path)
#disease_age_and_sex_analysis(df)
#disease_co_occurrence(df)
#disease_view_position_analysis(df)
#age_and_gender_distribution(df)
image_variation(df)
#age_vs_amount_of_diseases(df)