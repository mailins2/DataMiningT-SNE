import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('Data\customer_São_Paulo_2024.csv', encoding='utf-8')

missing_data_rows = df[df.isnull().any(axis=1)]
print("Dòng có giá trị thiếu:")
print(missing_data_rows)

# Xử lí giá trị bị thiếu 
# Xử lí các cột là giá trị số 
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
original_data = df[numeric_cols].copy()

for target_col in numeric_cols:
    if target_col in df.columns:
        col_mean = df[target_col].mean() 
        
        for i in range(len(df)):
            if pd.isnull(df.loc[i, target_col]):  
                pre_val = df.loc[i - 1, target_col] if i > 0 else np.nan 
                next_val = df.loc[i + 1, target_col] if i < len(df) - 1 else np.nan 

                if not pd.isnull(pre_val) and not pd.isnull(next_val):  
                    new_value = (pre_val + next_val) / 2
                else:  
                    new_value = col_mean

                df.loc[i, target_col] = round(new_value)



# Xử lý cột kiểu danh mục 
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].fillna(df[col].mode()[0])  # Điền giá trị phổ biến nhất (mode)

# Xử lý cột nhị phân (Binary Columns)
if 'SubscriptionStatus' in df.columns:
    existing_values = df['SubscriptionStatus'].dropna().unique()
    # Điền giá trị random vào các ô bị thiếu
    df['SubscriptionStatus'] = df['SubscriptionStatus'].apply(
        lambda x: np.random.choice(existing_values) if pd.isnull(x) else x)


# Kiểm tra có giá trị sai với yêu cầu mô tả 
def check_invalid_data(df):
    checks = {
        'Gender': lambda x: x not in ['Male', 'Female'],
        'Income': lambda x: pd.isnull(x) or x <= 0,
        'SpendingScore': lambda x: pd.isnull(x) or not (1 <= x <= 100),
        'EducationLevel': lambda x: x not in ['High School', 'Bachelor', 'Master', 'PhD'],
        'MaritalStatus': lambda x: x not in ['Single', 'Married', 'Divorced'],
        'PurchaseFrequency': lambda x: pd.isnull(x) or x < 0,
        'ProductCategory': lambda x: x not in ['Electronics', 'Fashion', 'Groceries', 'Furniture', 'Sports', 'Beauty'],
        'LoyaltyScore': lambda x: pd.isnull(x) or not (1 <= x <= 100),
        'EmploymentStatus': lambda x: x not in ['Employed', 'Unemployed', 'Student', 'Retired'],
        'HouseholdSize': lambda x: pd.isnull(x) or not (1 <= x <= 5),
        'CreditScore': lambda x: pd.isnull(x) or not (300 <= x <= 850),
        'OnlineShoppingHabit': lambda x: x not in ['Low', 'Medium', 'High'],
        'DiscountSensitivity': lambda x: x not in ['Low', 'Medium', 'High'],
        'PreferredPaymentMethod': lambda x: x not in ['Credit Card', 'PayPal', 'Cash', 'Installments'],
        'SubscriptionStatus': lambda x: x not in ['Yes', 'No'],
    }

    for col, condition in checks.items():
        if col in df.columns:
            invalid_rows = df[df[col].apply(condition)]
            print(f"\nCột `{col}` có {len(invalid_rows)} giá trị sai hoặc thiếu:")
            if not invalid_rows.empty:
                print(invalid_rows[[col]])
check_invalid_data(df)


# # Xử lí dữ liệu theo yêu cầu mô tả 
# Income 
df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
# Tính trung bình các giá trị Income hợp lệ (> 0)
valid_income_mean = df.loc[df['Income'] > 0, 'Income'].mean()
df['Income'] = df['Income'].apply(lambda x: x if pd.notnull(x) and x > 0 else round(valid_income_mean))
print("Sau khi xử lí Income")
check_invalid_data(df)

# Xuat file sau khi điền các giá trị bị thiếu va sai
df.to_csv("Data\pre-normalized_data.csv", index=False)

# Chuẩn hóa
# Xóa cột ID 
if 'CustomerID' in df.columns:
    df = df.drop(['CustomerID'], axis=1)
else:
    print("Cột 'CustomerID' không tồn tại trong DataFrame.")

#chuyen đổi Object
ord_cols ={
    'EducationLevel': ['High School', 'Bachelor', 'Master', 'PhD'],
    'DiscountSensitivity': ['Low', 'Medium', 'High'],
    'OnlineShoppingHabit': ['Low', 'Medium', 'High'],
    'Gender' : ['Male','Female'],
    'MaritalStatus':['Single','Married','Divorced'],
    'ProductCategory': ['Electronics','Fashion','Groceries','Furniture','Sports','Beauty'],
    'EmploymentStatus':['Employed','Unemployed','Student','Retired'],
    'PreferredPaymentMethod':['Credit Card','PayPal','Cash','Installments'],
}

for col , order in ord_cols.items():
    if col in df.columns:
        df[col]=pd.Categorical(df[col],categories=order, ordered=True).codes

# #chuyển đổi Binary
df['SubscriptionStatus']= df['SubscriptionStatus'].map({'Yes':1,'No':0})

# Chuyển đổi cột số 
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df.to_csv("Data\post-normalization_data.csv", index=False)