import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ===================== CLEAN DATA =====================
def clean_data(df):
    df = df.drop_duplicates()

    keep_columns = [
        'work_year', 'experience_level', 'employment_type', 'job_title',
        'salary_in_usd', 'employee_residence', 'remote_ratio',
        'company_location', 'company_size'
    ]
    df = df[keep_columns]
    df = df[df['salary_in_usd'] <= 350000]
    return df


# ===================== ENCODE FEATURES =====================
def encode_features(df, encoder=None, scaler=None, fit_encoder=True, fit_scaler=True):
    categorical_cols = [
        'experience_level', 'employment_type', 'job_title',
        'employee_residence', 'company_location', 'company_size'
    ]

    X_cat = df[categorical_cols]

    if fit_encoder:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_encoded = encoder.fit_transform(X_cat)
    else:
        X_encoded = encoder.transform(X_cat)

    encoded_df = pd.DataFrame(
        X_encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    numeric_cols = ['work_year', 'remote_ratio']
    X_numeric = df[numeric_cols]

    if fit_scaler:
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)
    else:
        X_numeric_scaled = scaler.transform(X_numeric)

    X_numeric_df = pd.DataFrame(
        X_numeric_scaled,
        columns=numeric_cols,
        index=df.index
    )

    X = pd.concat([X_numeric_df, encoded_df], axis=1)
    return X, encoder, scaler


# ===================== SELECT FEATURES =====================
def select_features(df):
    X = df.drop('salary_in_usd', axis=1)
    y = df['salary_in_usd']
    return X, y


# ===================== EDA PLOTS =====================
sns.set_theme(style="whitegrid", palette="muted")

# Biểu đồ 1: Histogram salary_in_usd
def plot_hist_salary(df):
    plt.figure(figsize=(8,5))
    plt.hist(df['salary_in_usd'], bins=50, edgecolor='black')
    plt.title("Histogram of Salary (USD)")
    plt.xlabel("Salary in USD")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# Biểu đồ 2: Boxplot salary_in_usd
def plot_box_salary(df):
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df['salary_in_usd'])
    plt.title("Boxplot of Salary (USD)")
    plt.xlabel("Salary in USD")
    plt.tight_layout()
    plt.show()


# Biểu đồ 3: Lương trung bình theo cấp độ kinh nghiệm
def plot_salary_by_experience(df):
    data = df.groupby('experience_level')['salary_in_usd'].mean().sort_values()

    plt.figure(figsize=(8,5))
    data.plot(kind='bar')
    plt.title("Average Salary by Experience Level")
    plt.xlabel("Experience Level")
    plt.ylabel("Average Salary (USD)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Biểu đồ 4: Lương trung bình theo quy mô công ty
def plot_salary_by_company_size(df):
    data = df.groupby('company_size')['salary_in_usd'].mean()

    plt.figure(figsize=(6,5))
    data.plot(kind='bar')
    plt.title("Average Salary by Company Size")
    plt.xlabel("Company Size")
    plt.ylabel("Average Salary (USD)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Biểu đồ 5: Pie chart top 10 job nhiều nhất
def plot_top_jobs_pie(df):
    top_jobs = df['job_title'].value_counts().head(10)

    plt.figure(figsize=(8,8))
    plt.pie(
        top_jobs,
        labels=top_jobs.index,
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title("Top 10 Job Titles by Frequency")
    plt.tight_layout()
    plt.show()


# Biểu đồ 6: Top 10 job có lương trung bình cao nhất
def plot_top_salary_jobs(df):
    data = (
        df.groupby('job_title')['salary_in_usd']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10,5))
    data.plot(kind='bar')
    plt.title("Top 10 Job Titles by Average Salary")
    plt.xlabel("Job Title")
    plt.ylabel("Average Salary (USD)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# Biểu đồ 7: Lương trung bình theo quốc gia công ty
def plot_salary_by_country(df):
    data = (
        df.groupby('company_location')['salary_in_usd']
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10,5))
    data.plot(kind='bar')
    plt.title("Average Salary by Company Location")
    plt.xlabel("Company Location")
    plt.ylabel("Average Salary (USD)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Biểu đồ 8: Heatmap lương theo kinh nghiệm & công việc
# Style cho đẹp, giống hình
sns.set_theme(style="white")

def plot_salary_heatmap(df):
    # Chỉ lấy top 20 job phổ biến để biểu đồ không quá rối
    top_jobs = df['job_title'].value_counts().head(20).index

    df_filtered = df[df['job_title'].isin(top_jobs)]

    # Tạo bảng pivot: Job Title x Experience Level
    pivot_table = pd.pivot_table(
        df_filtered,
        values='salary_in_usd',
        index='job_title',
        columns='experience_level',
        aggfunc='mean'
    )

    # Sắp xếp thứ tự level kinh nghiệm
    exp_order = ['EN', 'MI', 'SE', 'EX']
    pivot_table = pivot_table[exp_order]

    # Vẽ heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        pivot_table,
        annot=True,                # Hiển thị số trên ô
        fmt=".0f",                 # Không hiển thị số thập phân
        cmap="YlGnBu",             # Bảng màu giống hình
        linewidths=0.5,
        cbar_kws={'label': 'Average Salary (USD)'}
    )

    plt.title("Heatmap lương trung bình theo Job Title và Experience Level", fontsize=14)
    plt.xlabel("Trình độ kinh nghiệm (Experience Level)")
    plt.ylabel("Công việc (Job Title)")
    plt.tight_layout()
    plt.show()

# ===================== BIỂU ĐỒ 9 =====================
# Heatmap: work_year x company_size x salary_in_usd
def plot_salary_by_year_company_size(df):
    pivot = pd.pivot_table(
        df,
        values='salary_in_usd',
        index='company_size',
        columns='work_year',
        aggfunc='mean'
    )

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={'label': 'Average Salary (USD)'}
    )

    plt.title("Heatmap lương trung bình theo năm làm việc và quy mô công ty")
    plt.xlabel("Năm làm việc (Work Year)")
    plt.ylabel("Quy mô công ty (Company Size)")
    plt.tight_layout()
    plt.show()


# ===================== BIỂU ĐỒ 10 =====================
# Heatmap: job_title x company_size x salary_in_usd
def plot_salary_by_job_company_size(df):
    top_jobs = df['job_title'].value_counts().head(20).index
    df_filtered = df[df['job_title'].isin(top_jobs)]

    pivot = pd.pivot_table(
        df_filtered,
        values='salary_in_usd',
        index='job_title',
        columns='company_size',
        aggfunc='mean'
    )

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={'label': 'Average Salary (USD)'}
    )

    plt.title("Heatmap lương trung bình theo công việc và quy mô công ty")
    plt.xlabel("Quy mô công ty (Company Size)")
    plt.ylabel("Công việc (Job Title)")
    plt.tight_layout()
    plt.show()


# ===================== BIỂU ĐỒ 11 =====================
# Boxplot: employment_type x salary_in_usd
def plot_salary_by_employment_type(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x='employment_type',
        y='salary_in_usd',
        data=df
    )

    plt.title("Boxplot lương theo loại hợp đồng")
    plt.xlabel("Loại hợp đồng (Employment Type)")
    plt.ylabel("Salary in USD")
    plt.tight_layout()
    plt.show()


# ===================== BIỂU ĐỒ 12 =====================
# Heatmap: employment_type x company_size x salary_in_usd
def plot_salary_by_employment_company_size(df):
    pivot = pd.pivot_table(
        df,
        values='salary_in_usd',
        index='employment_type',
        columns='company_size',
        aggfunc='mean'
    )

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={'label': 'Average Salary (USD)'}
    )

    plt.title("Heatmap lương trung bình theo loại hợp đồng và quy mô công ty")
    plt.xlabel("Quy mô công ty (Company Size)")
    plt.ylabel("Loại hợp đồng (Employment Type)")
    plt.tight_layout()
    plt.show()

df = pd.read_csv("..\data\Data_salary.csv")
df = clean_data(df)

plot_hist_salary(df)
plot_box_salary(df)
plot_salary_by_experience(df)
plot_salary_by_company_size(df)
plot_top_jobs_pie(df)
plot_top_salary_jobs(df)
plot_salary_by_country(df)
plot_salary_heatmap(df)
plot_salary_by_year_company_size(df)       # Biểu đồ 9
plot_salary_by_job_company_size(df)        # Biểu đồ 10
plot_salary_by_employment_type(df)         # Biểu đồ 11
plot_salary_by_employment_company_size(df) # Biểu đồ 12
