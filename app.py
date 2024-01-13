import pandas as pd
import chayne
from chayne import infostock
import streamlit as st

import pandas as pd
import streamlit as st


# Define your functions here (df, dfb, infostock)

# Load your dataframes here (df2018, df2019, df2020, df2021, df2022)
dfinfo = pd.read_excel('Price-Vol VN 2015-2023.xlsx', sheet_name="Info")
dfprice = pd.read_excel('Price-Vol VN 2015-2023.xlsx', sheet_name="Price")
dfvolume = pd.read_excel('Price-Vol VN 2015-2023.xlsx', sheet_name="Volume")
df2018 = pd.read_excel('2018-Vietnam.xlsx')
df2019 = pd.read_excel('2019-Vietnam.xlsx')
df2020 = pd.read_excel('2020-Vietnam.xlsx')
df2021 = pd.read_excel('2021-Vietnam.xlsx')
df2022 = pd.read_excel('2022-Vietnam.xlsx')
@st.cache_data
def preprocess_dfs(df2018, df2019, df2020, df2021, df2022):
    dfs = [df2018.copy(), df2019.copy(), df2020.copy(), df2021.copy(), df2022.copy()]

    # Standardize column names
    column_replacements = [
        ('\nHợp nhất\nQuý: Hàng năm\nNăm:', ''),
        ('\nĐơn vị: Triệu VND', '')
    ]
    for df in dfs:
        df.columns = [col.replace(old, new) for col in df.columns for old, new in column_replacements]

    # Drop unnecessary columns
    columns_to_drop = [4, 5, 6, 8, 9, 10]
    for df in dfs:
        df.drop(df.columns[columns_to_drop], axis=1, inplace=True)

    # Concatenate DataFrames
    combined_df = pd.concat(dfs)

    # Sort and reset index
    combined_df.sort_values('Mã', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.columns = combined_df.columns.astype(str)
    combined_df.rename(columns={'STT': 'STT1'}, inplace=True)

    return combined_df

def infostock(symbol_to_lookup, df2018, df2019, df2020, df2021, df2022):
    # Preprocess DataFrames
    processed_dfs = preprocess_dfs(df2018, df2019, df2020, df2021, df2022)

    result_info = processed_dfs[0][processed_dfs[0]["Code"] == symbol_to_lookup]
    result_2018 = processed_dfs[0][processed_dfs[0]['Mã'] == symbol_to_lookup]

    industry = result_2018["Ngành ICB - cấp 4"].values[0]
    selected_companies = processed_dfs[0][processed_dfs[0]['Ngành ICB - cấp 4'] == industry]['Mã'].values

    company_name = result_info["Name"].values[0]
    stock_industry = result_info["Sector"].values[0]
    start_date = result_info['Start Date'].values[0]
    exchange = result_2018['Sàn'].values[0]
    ma_nganh = result_2018['Ngành ICB - cấp 4'].values[0]

    return company_name, stock_industry, start_date, exchange, ma_nganh, selected_companies


def preprocess_dfb(dfinfo, dfprice, dfvolume):
    # Make copies of the input dataframes
    dfinfo_copy = dfinfo.copy()
    dfprice_copy = dfprice.copy()
    dfvolume_copy = dfvolume.copy()

    # Rename 'Symbol' column to 'Code' in dfinfo_copy
    dfinfo_copy = dfinfo_copy.rename(columns={'Symbol': "Code"})

    # Remove specified patterns in the 'Code' column of all dataframes in dfb
    dfb = [dfinfo_copy, dfprice_copy, dfvolume_copy]
    for df in dfb:
        df['Code'] = df['Code'].str.replace(r'VT:|\(VO\)|\(P\)', '', regex=True)

    # Sorting and resetting index for dfinfo_copy, dfprice_copy, dfvolume_copy
    for df in dfb:
        df.sort_values('Code', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.columns = df.columns.astype(str)
        df.drop(df.columns[2], axis=1, inplace=True)  # Dropping a specific column

    # Process df1 dataframe
    df1 = dfb[1].rename(columns={'Code': "Date"}).drop("Name", axis=1).transpose()
    df1.columns = df1.iloc[0]
    df1 = df1[1:].reset_index().rename(columns={'index': 'Date'})
    df1['Date'] = pd.to_datetime(df1['Date']).dt.strftime('%Y/%m/%d')
    df1 = df1.sort_values(by='Date', ascending=True)

    # Process df2 dataframe similar to df1 processing
    df2 = dfb[2].rename(columns={'Code': "Date"}).drop("Name", axis=1).transpose()
    df2.columns = df2.iloc[0]
    df2 = df2[1:].reset_index().rename(columns={'index': 'Date'})
    df2['Date'] = pd.to_datetime(df2['Date']).dt.strftime('%Y/%m/%d')
    df2 = df2.sort_values(by='Date', ascending=True)

    # Process in4_column dataframe
    in4_column = dfb[0][['Code', 'Start Date']].copy()
    in4_column['Start Date'] = pd.to_datetime(in4_column['Start Date']).dt.strftime('%Y/%m/%d')

    return dfinfo, dfprice, dfvolume,in4_column  # Returning processed dataframes


if __name__ == "__main__":
    st.title("Ứng dụng Thông tin cổ phiếu")
    symbol_to_lookup = st.text_input("Nhập mã cổ phiếu")
    button_clicked = st.button("Tìm kiếm")

    if button_clicked:
        if symbol_to_lookup:
            # Assuming you've loaded dfinfo_copy, dfvolume_copy, dfprice_copy, and df2018_copy earlier

            # Perform processing on the dataframes using preprocess_dfb before calling infostock
            dfinfo_processed, dfvolume_processed, dfprice_processed = preprocess_dfb(dfinfo, dfprice, dfvolume)

            # Use infostock with the processed dataframes
            company_name, industry, start_date, exchange, ma_nganh, selected_companies = infostock(symbol_to_lookup, dfinfo_processed, dfvolume_processed, dfprice_processed, df2018_copy)

            st.title("Thông tin cổ phiếu")
            st.subheader(f"Thông tin của mã cổ phiếu {symbol_to_lookup}:")
            st.write(f"Tên công ty: {company_name}")
            st.write(f"Ngành công nghiệp: {industry}")
            st.write(f"Ngày bắt đầu giao dịch: {start_date}")
            st.write(f"Sàn: {exchange}")
            st.write(f"Mã ngành: {ma_nganh}")
            st.subheader("Các mã cùng ngành:")
            st.write(selected_companies)

