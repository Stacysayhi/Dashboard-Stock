import numpy as np
import pandas as pd
import seaborn as sn
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import tabulate
from tabulate import tabulate
import re
import matplotlib.pyplot as plt

dfinfo = pd.read_excel('Price-Vol VN 2015-2023.xlsx', sheet_name="Info")
dfprice = pd.read_excel('Price-Vol VN 2015-2023.xlsx', sheet_name="Price")
dfvolume = pd.read_excel('Price-Vol VN 2015-2023.xlsx', sheet_name="Volume")
df2018 = pd.read_excel('2018-Vietnam.xlsx')
df2019 = pd.read_excel('2019-Vietnam.xlsx')
df2020 = pd.read_excel('2020-Vietnam.xlsx')
df2021 = pd.read_excel('2021-Vietnam.xlsx')
df2022 = pd.read_excel('2022-Vietnam.xlsx')


def df():
    df2018_copy, df2019_copy, df2020_copy, df2021_copy, df2022_copy = df2018.copy(), df2019.copy(), df2020.copy(), df2021.copy(), df2022.copy()

    dfa = [df2018_copy, df2019_copy, df2020_copy, df2021_copy, df2022_copy]

    # Chuẩn hóa việc đổi tên cột
    column_replacements = [
        ('\nHợp nhất\nQuý: Hàng năm\nNăm:', ''),
        ('\nĐơn vị: Triệu VND', '')
    ]
    for df in dfa:
        df.columns = [col.replace(old, new) for col in df.columns for old, new in column_replacements]

    # Loại bỏ các cột không cần thiết
    columns_to_drop = [4, 5, 6, 8, 9, 10]
    for df in dfa:
        df.drop(df.columns[columns_to_drop], axis=1, inplace=True)

    # Ghép nối DataFrame
    df = pd.concat(dfa)

    df.sort_values('Mã', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.columns = df.columns.astype(str)  # Chuyển đổi tên cột thành kiểu dữ liệu chuỗi
    df.rename(columns={'STT': 'STT1'}, inplace=True)

    return df2018_copy, df2019_copy, df2020_copy, df2021_copy, df2022_copy

def dfb(dfinfo, dfprice, dfvolume):
    dfinfo_copy = dfinfo.copy()
    dfprice_copy = dfprice.copy()
    dfvolume_copy = dfvolume.copy()
    dfinfo_copy = dfinfo_copy.rename(columns={'Symbol': "Code"})
    dfb = [dfinfo_copy, dfprice_copy, dfvolume_copy]
    for df in dfb:
        df['Code'] = df['Code'].str.replace(r'VT:|\(VO\)|\(P\)', '', regex=True)
    dfinfo_copy.sort_values('Code', inplace=True)
    dfinfo_copy.reset_index(drop=True, inplace=True)
    dfinfo_copy.columns = dfinfo_copy.columns.astype(str)
    dfinfo_copy.drop(dfinfo_copy.columns[2], axis=1, inplace=True)
    dfprice_copy.sort_values('Code', inplace=True)
    dfprice_copy.reset_index(drop=True, inplace=True)
    dfprice_copy.columns = dfprice_copy.columns.astype(str)
    dfprice_copy.drop(dfprice_copy.columns[2], axis=1, inplace=True)
    dfvolume_copy.sort_values('Code', inplace=True)
    dfvolume_copy.reset_index(drop=True, inplace=True)
    dfvolume_copy.columns = dfvolume_copy.columns.astype(str)
    dfvolume_copy.drop(dfvolume_copy.columns[2], axis=1, inplace=True)
    df1 = dfb[1]
    df2 = dfb[2]
    df3 = dfb[0]
    df1 = df1.rename(columns={'Code': "Date"})
    df1 = df1.drop("Name", axis=1)
    df1 = df1.transpose()
    new_header = df1.iloc[0]
    df1 = df1[1:]
    df1.columns = new_header
    df1 = df1.reset_index()
    df1 = df1.rename(columns={'index': 'Date'})
    df1 = df1.sort_values(by='Date', ascending=True)
    df1['Date'] = pd.to_datetime(df1['Date']).dt.strftime('%Y/%m/%d')

    df2 = df2.rename(columns={'Code': "Date"})
    df2 = df2.drop("Name", axis=1)
    df2 = df2.transpose()
    new_header = df2.iloc[0]
    df2 = df2[1:]
    df2.columns = new_header
    df2 = df2.reset_index()
    df2 = df2.rename(columns={'index': 'Date'})
    df2 = df2.sort_values(by='Date', ascending=True)
    df2['Date'] = pd.to_datetime(df2['Date']).dt.strftime('%Y/%m/%d')
    in4_column = df3[['Code', 'Start Date']].copy()
    in4_column['Start Date'] = pd.to_datetime(in4_column['Start Date']).dt.strftime('%Y/%m/%d')
    return df3, df2, df1

def infostock(symbol_to_lookup, dfinfo_copy, dfvolume_copy, dfprice_copy, df2018_copy):
    df2019_copy, df2020_copy, df2021_copy, df2022_copy = df2018_copy.copy(), df2018_copy.copy(), df2018_copy.copy(), df2018_copy.copy()
    dfinfo_copy.sort_values('Code', inplace=True)
    dfinfo_copy.reset_index(drop=True, inplace=True)
    dfvolume_copy.sort_values('Code', inplace=True)
    dfvolume_copy.reset_index(drop=True, inplace=True)
    dfprice_copy.sort_values('Code', inplace=True)
    dfprice_copy.reset_index(drop=True, inplace=True)

    result = dfinfo_copy[dfinfo_copy["Code"] == symbol_to_lookup]
    result1 = df2018_copy[df2018_copy['Mã'] == symbol_to_lookup]

    industry = result1["Ngành ICB - cấp 4"].values[0]

    selected_companies = df2018_copy[df2018_copy['Ngành ICB - cấp 4'] == industry]['Mã'].values

    company_name = result["Name"].values[0]
    industry = result["Sector"].values[0]
    start_date = result['Start Date'].values[0]
    exchange = result1['Sàn'].values[0]
    ma_nganh = result1['Ngành ICB - cấp 4'].values[0]

    return company_name, industry, start_date, exchange, ma_nganh, selected_companies

def giacophieu(symbol_to_lookup, in4_column, df1, df2, start_date):
    # Tạo danh sách thông báo để trả về
    messages = []

    # Gọi hàm dfb() và gán tất cả giá trị trả về cho biến result
    result = dfb()

    # Truy xuất các giá trị từ biến result
    dfinfo_copy, dfvolume_copy, dfprice_copy, in4_column_copy = result

    filtered_df = in4_column_copy.loc[in4_column_copy['Code'] == symbol_to_lookup]

    if not filtered_df.empty:
        # Truy cập cột 'Start Date' của DataFrame lọc
        start_date = filtered_df['Start Date'].values[0]

        # Lọc các hàng từ 'start date' trở về sau trong DataFrame 'df1'
        filtered_date_df = df1.loc[df1['Date'] >= start_date]

        # Kiểm tra nếu DataFrame lọc không rỗng
        if not filtered_date_df.empty:
            # Kiểm tra tên cột 'symbol_to_lookup' có tồn tại trong DataFrame lọc hay không
            if symbol_to_lookup in filtered_date_df.columns:
                # Truy cập cột 'symbol_to_lookup' của DataFrame lọc
                close_price = filtered_date_df[symbol_to_lookup]

                # Tạo DataFrame mới từ cột 'Start Date' và 'symbol_to_lookup'
                merged_df = pd.DataFrame({'Start Date': filtered_date_df['Date'], symbol_to_lookup: close_price})

                # Lọc các ngày có khối lượng giao dịch
                filtered_volume_df = df2.loc[df2[symbol_to_lookup] > 0]

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=merged_df['Start Date'], y=merged_df[symbol_to_lookup], mode='lines',
                                         name='Giá đóng cửa', line=dict(color='blue'), connectgaps=True))

                fig.update_layout(
                    title=f"Biểu đồ giá đóng cửa của cổ phiếu {symbol_to_lookup}",
                    xaxis=dict(title='Ngày'),
                    yaxis=dict(title='Giá đóng cửa'),
                )

                # Lấy chỉ mục của các ngày có khối lượng giao dịch khác null
                indices = filtered_volume_df[symbol_to_lookup].notnull()

                fig.add_trace(
                    go.Bar(x=filtered_volume_df['Date'][indices], name='Khối lượng', marker=dict(color='green')))

                fig.update_layout(
                    title=f"Biểu đồ giá đóng cửa và khối lượng giao dịch của cổ phiếu {symbol_to_lookup}",
                    xaxis=dict(title='Ngày'),
                )

                return fig

            else:
                messages.append(f"Không tìm thấy cột '{symbol_to_lookup}' trong DataFrame lọc.")
        else:
            messages.append(f"Không tìm thấy dữ liệu từ '{start_date}' trở đi trong DataFrame 'df1'.")
    else:
        messages.append(f"Không tìm thấy mã '{symbol_to_lookup}' trong DataFrame 'in4_column'.")

    return messages, None


cac_bao_cao = ['CĐKT', 'KQKD', 'LCTT', 'TM BCTT', 'BCTCKH']
def cacbaocao(symbol_to_lookup):
    cac_bao_cao = ['CĐKT', 'KQKD', 'LCTT', 'TM BCTT', 'BCTCKH']
    years = ['2018', '2019', '2020', '2021', '2022']
    if cac_bao_cao== 'CĐKT':
        print(f'Bảng cân đối kế toán của mã {symbol_to_lookup}')
        selected_years = years
        df_concat = pd.DataFrame()

        for year in selected_years:
            df_filtered = globals()[f'df{year}_copy'].loc[globals()[f'df{year}_copy']['Mã'] == symbol_to_lookup]
            cdkt_columns = df_filtered.filter(regex=r'^CĐKT', axis=1)
            df_melted = cdkt_columns.reset_index().melt(id_vars=['index'], var_name='Cột', value_name='Giá trị')
            df_concat = pd.concat([df_concat, df_melted['Giá trị']], axis=1)
            df_melted['Tên gọi'] = df_melted['Cột'].apply(lambda x: re.sub(r'2022', '', x).strip())
            df_melted['Tên gọi'] = df_melted['Tên gọi'].str.replace(r'^CĐKT\.', '', regex=True).str.strip()

        df_concat.columns = selected_years
        df_concat = pd.concat([df_melted['Tên gọi'], df_concat], axis=1)
        # Tính toán kích thước cột
        num_columns = len(df_concat.columns)
        column_width = 10 / num_columns  # Kích thước của mỗi cột (cm)

        # Tính toán kích thước dòng
        table_height = 8  # Chiều cao của bảng (cm)
        row_height = table_height / len(df_concat)  # Kích thước của mỗi dòng (cm)

        # Chuyển đổi kích thước bảng thành pixel
        scale_factor = 4  # Tỉ lệ chuyển đổi từ cm sang pixel (tùy chọn)
        table_width_px = num_columns * column_width * scale_factor
        table_height_px = len(df_concat) * row_height * scale_factor
        from tabulate import tabulate
        # In bảng với kích thước tùy chỉnh
        table = tabulate(df_concat, headers='keys', tablefmt='fancy_grid', showindex=False, numalign='center',
                         stralign='center', floatfmt=".1f")
        print(table)
    elif cac_bao_cao=='KQKD':
        print(f'Bảng báo cáo kết quả kinh doanh của mã {symbol_to_lookup}')
        selected_years = years
        df_concat = pd.DataFrame()
        for year in selected_years:
            df_filtered = globals()[f'df{year}_copy'].loc[globals()[f'df{year}_copy']['Mã'] == symbol_to_lookup]
            # Lọc các cột có tên bắt đầu bằng "CĐKT"
            kqkd_columns = df_filtered.filter(regex=r'^KQKD', axis=1)

            # Sử dụng phương thức melt để biến đổi DataFrame
            df_melted = kqkd_columns.reset_index().melt(id_vars=['index'], var_name='Cột', value_name='Giá trị')

            # Sắp xếp lại DataFrame theo thứ tự tên cột nằm theo hàng dọc
            df_concat = pd.concat([df_concat, df_melted['Giá trị']], axis=1)
            df_melted['Tên gọi'] = df_melted['Cột']  # Add the column with the names
            df_melted['Tên gọi'] = df_melted['Cột'].apply(lambda x: re.sub(r'2022', '', x).strip())
        df_concat.columns = selected_years
        df_concat = pd.concat([df_melted['Tên gọi'], df_concat], axis=1)
        # In ra bảng đã sắp xếp
        from tabulate import tabulate

        # Chuyển đổi DataFrame thành danh sách các hàng
        table_data = df_concat.values.tolist()

        # Lấy danh sách tên cột
        headers = df_concat.columns.tolist()

        # Đóng khung bảng và in ra
        table = tabulate(table_data, headers, tablefmt='fancy_grid')
        print(table)
    elif cac_bao_cao == 'LCTT':
        print(f'Bảng lưu chuyển tiền tệ của mã {symbol_to_lookup}')
        selected_years = years
        df_concat = pd.DataFrame()
        for year in selected_years:
            df_filtered = globals()[f'df{year}_copy'].loc[globals()[f'df{year}_copy']['Mã'] == symbol_to_lookup]
            # Lọc các cột có tên bắt đầu bằng "CĐKT"
            lctt_columns = df_filtered.filter(regex=r'^LCTT', axis=1)

            # Sử dụng phương thức melt để biến đổi DataFrame
            df_melted = lctt_columns.reset_index().melt(id_vars=['index'], var_name='Cột', value_name='Giá trị')

            df_concat = pd.concat([df_concat, df_melted['Giá trị']], axis=1)
            df_melted['Tên gọi'] = df_melted['Cột']  # Add the column with the names
            df_melted['Tên gọi'] = df_melted['Cột'].apply(lambda x: re.sub(r'2022', '', x).strip())
        df_concat.columns = selected_years
        df_concat = pd.concat([df_melted['Tên gọi'], df_concat], axis=1)

        # In ra bảng đã sắp xếp
        from tabulate import tabulate

        # Chuyển đổi DataFrame thành danh sách các hàng
        table_data = df_concat.values.tolist()

        # Lấy danh sách tên cột
        headers = df_concat.columns.tolist()

        # Đóng khung bảng và in ra
        table = tabulate(table_data, headers, tablefmt='fancy_grid')
        print(table)
    elif cac_bao_cao=='TM BCTT':
        print(f'Bảng thuyết minh báo cáo tiền tệ của mã {symbol_to_lookup}')
        selected_years = years
        df_concat = pd.DataFrame()
        for year in selected_years:
            df_filtered = globals()[f'df{year}_copy'].loc[globals()[f'df{year}_copy']['Mã'] == symbol_to_lookup]
            # Lọc các cột có tên bắt đầu bằng "CĐKT"
            tm_columns = df_filtered.filter(regex=r'^TM', axis=1)

            # Sử dụng phương thức melt để biến đổi DataFrame
            df_melted = tm_columns.reset_index().melt(id_vars=['index'], var_name='Cột', value_name='Giá trị')

            df_concat = pd.concat([df_concat, df_melted['Giá trị']], axis=1)
            df_melted['Tên gọi'] = df_melted['Cột']  # Add the column with the names
            df_melted['Tên gọi'] = df_melted['Cột'].apply(lambda x: re.sub(r'2022', '', x).strip())

        df_concat.columns = selected_years
        df_concat = pd.concat([df_melted['Tên gọi'], df_concat], axis=1)

        # In ra bảng đã sắp xếp
        from tabulate import tabulate

        # Chuyển đổi DataFrame thành danh sách các hàng
        table_data = df_concat.values.tolist()

        # Lấy danh sách tên cột
        headers = df_concat.columns.tolist()

        # Đóng khung bảng và in ra
        table = tabulate(table_data, headers, tablefmt='fancy_grid')
        print(table)
    else:
        print(f'Bảng báo cáo tài chính kế hoạch của mã {symbol_to_lookup}')
        selected_years = years
        df_concat = pd.DataFrame()
        for year in selected_years:
            df_filtered = globals()[f'df{year}_copy'].loc[globals()[f'df{year}_copy']['Mã'] == symbol_to_lookup]
            # Lọc các cột có tên bắt đầu bằng "CĐKT"
            bctckh_columns = df_filtered.filter(regex=r'^BCTCKH', axis=1)

            # Sử dụng phương thức melt để biến đổi DataFrame
            df_melted = bctckh_columns.reset_index().melt(id_vars=['index'], var_name='Cột', value_name='Giá trị')

            df_concat = pd.concat([df_concat, df_melted['Giá trị']], axis=1)
            df_melted['Tên gọi'] = df_melted['Cột']  # Add the column with the names
            df_melted['Tên gọi'] = df_melted['Cột'].apply(lambda x: re.sub(r'2022', '', x).strip())
        df_concat.columns = selected_years
        df_concat = pd.concat([df_melted['Tên gọi'], df_concat], axis=1)

        # In ra bảng đã sắp xếp
        from tabulate import tabulate

        # Chuyển đổi DataFrame thành danh sách các hàng
        table_data = df_concat.values.tolist()

        # Lấy danh sách tên cột
        headers = df_concat.columns.tolist()

        # Đóng khung bảng và in ra
        table = tabulate(table_data, headers, tablefmt='fancy_grid')
        print(table)
def cactysotaichinh(symbol_to_lookup):
    ti_so = ['Tỷ số thanh khoản', 'Tỷ số quản lý tài sản', 'Tỷ số quản lý nợ', 'Tỷ số sinh lợi']
    if cactysotaichinh=='Tỷ số thanh khoản':
        b = df.loc[df['Mã'] == symbol_to_lookup, 'Mã'].values[0]

        start_year = 2018
        end_year = 2022

        selected_years = range(start_year, end_year + 1)
        current_ratios = []
        quick_ratios = []

        for year in selected_years:
            try:
                current_ratio = tinh_ty_so_thanh_toan_hien_hanh(
                    df.loc[df['Mã'] == b, f'CĐKT. TÀI SẢN NGẮN HẠN {year}'].values[0],
                    df.loc[df['Mã'] == b, f'CĐKT. TỔNG CỘNG TÀI SẢN {year}'].values[0])
                quick_ratio = tinh_ty_so_thanh_toan_nhanh(
                    df.loc[df['Mã'] == b, f'CĐKT. TÀI SẢN NGẮN HẠN {year}'].values[0],
                    df.loc[df['Mã'] == b, f'CĐKT. Hàng tồn kho, ròng {year}'].values[0],
                    df.loc[df['Mã'] == b, f'CĐKT. TỔNG CỘNG TÀI SẢN {year}'].values[0])

                current_ratios.append(current_ratio)
                quick_ratios.append(quick_ratio)

            except IndexError:
                print(f"Giá trị {b} không tồn tại trong cột 'Mã'.")

        average_current_ratios = []
        average_quick_ratios = []
        selected_companies=infostock()
        for year in selected_years:
            current_ratios_per_year = []
            quick_ratios_per_year = []

            for company in selected_companies:
                try:
                    current_asset_short_term = df.loc[df['Mã'] == company, f'CĐKT. TÀI SẢN NGẮN HẠN {year}'].values[0]
                    total_assets = df.loc[df['Mã'] == company, f'CĐKT. TỔNG CỘNG TÀI SẢN {year}'].values[0]
                    inventory = df.loc[df['Mã'] == company, f'CĐKT. Hàng tồn kho, ròng {year}'].values[0]

                    if np.isnan(current_asset_short_term) or current_asset_short_term == 0 or np.isnan(
                            total_assets) or total_assets == 0:
                        continue

                    current_ratio = tinh_ty_so_thanh_toan_hien_hanh(current_asset_short_term, total_assets)
                    quick_ratio = tinh_ty_so_thanh_toan_nhanh(current_asset_short_term, inventory, total_assets)

                    if np.isnan(current_ratio) or np.isnan(quick_ratio):
                        continue

                    current_ratios_per_year.append(current_ratio)
                    quick_ratios_per_year.append(quick_ratio)
                except IndexError:
                    # Xử lý ngoại lệ nếu không tìm thấy dữ liệu cho năm hoặc mã cụ thể
                    pass

            if current_ratios_per_year and quick_ratios_per_year:
                current_ratio_avg = sum(current_ratios_per_year) / len(current_ratios_per_year)
                quick_ratio_avg = sum(quick_ratios_per_year) / len(quick_ratios_per_year)

                average_current_ratios.append(current_ratio_avg)
                average_quick_ratios.append(quick_ratio_avg)

                # Định nghĩa chiều rộng của mỗi cột
        bar_width = 0.2

        # Tạo mảng chỉ số cho các cột
        index = np.arange(len(selected_years))

        # Vẽ biểu đồ cột
        plt.figure(figsize=(8, 6))  # Điều chỉnh kích thước của biểu đồ theo ý muốn
        plt.bar(index, current_ratios, bar_width, label='Current Ratio', color='blue')
        plt.bar(index + bar_width, average_current_ratios, bar_width, label='Average Current Ratios', color='green')
        plt.bar(index + 2 * bar_width, quick_ratios, bar_width, label='Quick Ratio', color='red')
        plt.bar(index + 3 * bar_width, average_quick_ratios, bar_width, label='Average Quick Ratios', color='orange')

        plt.xlabel('Năm')
        plt.ylabel('Tỷ số thanh khoản')
        plt.title(f'Tỷ số thanh khoản của mã {b}')
        plt.xticks(index + bar_width, selected_years)
        plt.legend()
        plt.grid(True)  # Add grid lines for better readability
        plt.show()

        # Tạo DataFrame từ current_ratios và quick_ratios
        data = {'Năm': selected_years,
                'Current Ratio': current_ratios,
                'Quick Ratio': quick_ratios,
                'Average Industry Current Ratios': average_current_ratios,
                'Average Industry Quick Ratios': average_quick_ratios}

        df_table = pd.DataFrame(data)
        print(df_table.to_string(index=False))

    elif cactysotaichinh== "Tỷ số quản lý tài sản":
        b = df.loc[df['Mã'] == symbol_to_lookup, 'Mã'].values[0]

        start_year = 2018
        end_year = 2022

        selected_years = range(start_year, end_year + 1)
        inventory_ratios = []
        DSO_ratios = []
        FA_turnover_ratios = []
        TA_turnover_ratios = []

        for year in selected_years:
            try:
                AR = df.loc[df['Mã'] == b, f'CĐKT. Các khoản phải thu ngắn hạn {year}'].values[0] + \
                     df.loc[df['Mã'] == b, f'CĐKT. Phải thu dài hạn {year}'].values[0]

                inventory_ratio = vong_quay_hang_ton_kho(
                    df.loc[df['Mã'] == b, f'KQKD. Doanh thu thuần {year}'].values[0],
                    df.loc[df['Mã'] == b, f'CĐKT. Hàng tồn kho, ròng {year}'].values[0])
                DSO_ratio = ky_thu_tien_binh_quan(AR, df.loc[df['Mã'] == b, f'KQKD. Doanh thu thuần {year}'].values[0])

                inventory_ratios.append(inventory_ratio)
                DSO_ratios.append(DSO_ratio)

                FA_turnover_ratio = vong_quay_TSCD(df.loc[df['Mã'] == b, f'KQKD. Doanh thu thuần {year}'].values[0], (
                            df.loc[df['Mã'] == b, f'CĐKT. Tài sản cố định {year}'].values[0] -
                            df.loc[df['Mã'] == b, f'LCTT. Khấu hao TSCĐ   {year}'].values[0]))
                TA_turnover_ratio = vong_quay_TTS(df.loc[df['Mã'] == b, f'KQKD. Doanh thu thuần {year}'].values[0],
                                                  df.loc[df['Mã'] == b, f'CĐKT. TỔNG CỘNG TÀI SẢN {year}'].values[0])

                FA_turnover_ratios.append(FA_turnover_ratio)
                TA_turnover_ratios.append(TA_turnover_ratio)

            except IndexError:
                print(f"Giá trị {b} không tồn tại trong cột 'Mã'.")

        average_inventory_ratios = []
        average_DSO_ratios = []
        average_FAturnover_ratios = []
        average_TAturnover_ratios = []
        selected_companies = infostock()
        for year in selected_years:
            inventory_ratios_per_year = []
            DSO_ratios_per_year = []
            FAturnover_ratios_per_year = []
            TAturnover_ratios_per_year = []


            for company in selected_companies:
                try:
                    AR_short_term = df.loc[df['Mã'] == company, f'CĐKT. Các khoản phải thu ngắn hạn {year}'].values[0]
                    AR_long_term = df.loc[df['Mã'] == company, f'CĐKT. Phải thu dài hạn {year}'].values[0]
                    AR = AR_short_term + AR_long_term

                    if np.isnan(AR_short_term) or np.isnan(AR_long_term) or np.isnan(AR) or \
                            AR_short_term == 0 or AR_long_term == 0 or AR == 0:
                        continue

                    net_revenue = df.loc[df['Mã'] == company, f'KQKD. Doanh thu thuần {year}'].values[0]
                    inventory = df.loc[df['Mã'] == company, f'CĐKT. Hàng tồn kho, ròng {year}'].values[0]

                    if np.isnan(net_revenue) or np.isnan(inventory) or net_revenue == 0 or inventory == 0:
                        continue

                    inventory_ratio = vong_quay_hang_ton_kho(net_revenue, inventory)
                    DSO_ratio = ky_thu_tien_binh_quan(AR, net_revenue)
                    FA_turnover_ratio = vong_quay_TSCD(net_revenue, (
                                df.loc[df['Mã'] == company, f'CĐKT. Tài sản cố định {year}'].values[0] -
                                df.loc[df['Mã'] == company, f'LCTT. Khấu hao TSCĐ   {year}'].values[0]))
                    TA_turnover_ratio = vong_quay_TTS(net_revenue, df.loc[
                        df['Mã'] == company, f'CĐKT. TỔNG CỘNG TÀI SẢN {year}'].values[0])

                    if np.isnan(inventory_ratio) or np.isnan(DSO_ratio) or np.isnan(FA_turnover_ratio) or np.isnan(
                            TA_turnover_ratio) or inventory_ratio == 0 or DSO_ratio == 0 or FA_turnover_ratio == 0 or TA_turnover_ratio == 0:
                        continue

                    inventory_ratios_per_year.append(inventory_ratio)
                    DSO_ratios_per_year.append(DSO_ratio)
                    FAturnover_ratios_per_year.append(FA_turnover_ratio)
                    TAturnover_ratios_per_year.append(TA_turnover_ratio)

                except IndexError:
                    # Xử lý ngoại lệ nếu không tìm thấy dữ liệu cho năm hoặc mã cụ thể
                    pass

            if inventory_ratios_per_year:
                average_inventory_ratio = sum(inventory_ratios_per_year) / len(inventory_ratios_per_year)
                average_inventory_ratios.append(average_inventory_ratio)

            if DSO_ratios_per_year:
                average_DSO_ratio = sum(DSO_ratios_per_year) / len(DSO_ratios_per_year)
                average_DSO_ratios.append(average_DSO_ratio)

            if FAturnover_ratios_per_year:
                average_FAturnover_ratio = sum(FAturnover_ratios_per_year) / len(FAturnover_ratios_per_year)
                average_FAturnover_ratios.append(average_FAturnover_ratio)

            if TAturnover_ratios_per_year:
                average_TAturnover_ratio = sum(TAturnover_ratios_per_year) / len(TAturnover_ratios_per_year)
                average_TAturnover_ratios.append(average_TAturnover_ratio)

        data = {
            'Năm': selected_years,
            'Inventory Ratio': inventory_ratios,
            'DSO Ratio': DSO_ratios,
            'Fix Assets turn over': FA_turnover_ratios,
            'Total Assets turn over': TA_turnover_ratios,
            'Inventory Industry Ratio': average_inventory_ratios,
            'DSO Industry Ratio': average_DSO_ratios,
            'Fix Asets TO Industry': average_FAturnover_ratios,
            'Total Assets TO Industry': average_TAturnover_ratios
        }

        df_table1 = pd.DataFrame(data)

        # Định nghĩa chiều rộng của mỗi cột
        bar_width = 0.2

        # Tạo mảng chỉ số cho các cột
        index = np.arange(len(selected_years))

        # Vẽ biểu đồ cột
        plt.figure(figsize=(8, 6))  # Điều chỉnh kích thước của biểu đồ theo ý muốn
        plt.bar(index, inventory_ratios, bar_width, label='Inventory Ratio', color='blue')
        plt.bar(index + bar_width, average_inventory_ratios, bar_width, label='Inventory Industry Ratio', color='green')
        plt.bar(index + 2 * bar_width, DSO_ratios, bar_width, label='DSO Ratio', color='red')
        plt.bar(index + 3 * bar_width, average_DSO_ratios, bar_width, label='DSO Industry Ratio', color='orange')
        plt.xlabel('Năm')
        plt.ylabel('Tỷ số quản lý')
        plt.title(f'Tỷ số quản lý của mã {b}')
        plt.xticks(index + bar_width, selected_years)
        plt.legend()
        plt.grid(True)  # Add grid lines for better readability
        plt.show()

        plt.figure(figsize=(8, 6))  # Điều chỉnh kích thước của biểu đồ theo ý muốn
        plt.bar(index, FA_turnover_ratios, bar_width, label='Fix Assets turn over', color='blue')
        plt.bar(index + bar_width, average_FAturnover_ratios, bar_width, label='Fix Asets TO Industry', color='green')
        plt.bar(index + 2 * bar_width, TA_turnover_ratios, bar_width, label='Total Assets turn over', color='red')
        plt.bar(index + 3 * bar_width, average_TAturnover_ratios, bar_width, label='Total Assets TO Industry',
                color='orange')

        plt.xlabel('Năm')
        plt.ylabel('Tỷ số quản lý')
        plt.title(f'Tỷ số quản lý của mã {b}')
        plt.xticks(index + bar_width, selected_years)
        plt.legend()
        plt.grid(True)  # Add grid lines for better readability
        plt.show()
    elif cactysotaichinh=='Tỷ số quản lý nợ':
        b = df.loc[df['Mã'] == symbol_to_lookup, 'Mã'].values[0]

        start_year = 2018
        end_year = 2022

        selected_years = range(start_year, end_year + 1)
        debt_ratios = []
        TIE_ratios = []
        EBITDA_coverage_ratios = []
        for year in selected_years:
            try:
                Total_debt = df.loc[df['Mã'] == b, f'CĐKT. NỢ PHẢI TRẢ {year}'].values[0]
                TA = df.loc[df['Mã'] == b, f'CĐKT. TỔNG CỘNG TÀI SẢN {year}'].values[0]
                debt_ratio = ty_so_no_vay(Total_debt, TA)
                debt_ratios.append(debt_ratio)

                EBIT_value = EBIT()
                IE = df.loc[df['Mã'] == b, f'KQKD. Trong đó: Chi phí lãi vay {year}'].values[0]
                TIE_ratio = he_so_chi_tra_lai(EBIT_value, IE)
                TIE_ratios.append(TIE_ratio)

                EBITDA_value = EBITDA()
                tttienthue = \
                df.loc[df['Mã'] == b, f'KQKD. Lợi nhuận gộp về bán hàng và cung cấp dịch vụ {year}'].values[0]
                laivay = df.loc[df['Mã'] == b, f'CĐKT. NỢ PHẢI TRẢ {year}'].values[0]
                no_goc = df.loc[df['Mã'] == b, f'CĐKT. NỢ PHẢI TRẢ {year}'].values[0]
                EBITDA_coverage_ratio = he_so_kha_nang_tra_no(EBITDA_value, tttienthue, laivay, no_goc)
                EBITDA_coverage_ratios.append(EBITDA_coverage_ratio)

            except IndexError:
                print(f"Giá trị {b} không tồn tại trong cột 'Mã'.")

        data = {'Năm': selected_years, 'Debt Ratio': debt_ratios, 'TIE Ratio': TIE_ratios,
                'EBITDA Coverage Ratio': EBITDA_coverage_ratios}
        df_table = pd.DataFrame(data)

        # In bảng hiển thị kết quả
        print(df_table)

        # Tạo biểu đồ tỷ số nợ vay và tỷ số chi trả lãi
        plt.subplot(2, 1, 1)
        plt.plot(selected_years, debt_ratios, marker='o', label='Debt Ratio')
        plt.plot(selected_years, EBITDA_coverage_ratios, marker='o', label='TIE Ratio')
        plt.xlabel('Năm')
        plt.ylabel('Tỷ số thanh toán')
        plt.title(f'Biểu đồ tỷ số nợ vay và tỷ lệ khả năng trả nợ EBITDA của mã {b}')
        plt.xticks(range(start_year, end_year + 1))
        plt.legend()

        # Tạo biểu đồ tỷ lệ khả năng trả nợ EBITDA
        plt.subplot(2, 1, 2)
        plt.plot(selected_years, TIE_ratios, marker='o', label='EBITDA coverage Ratio')
        plt.xlabel('Năm')
        plt.ylabel('Tỷ số thanh toán')
        plt.title(f'Biểu đồ tỷ số trả lãi của mã {b}')
        plt.xticks(range(start_year, end_year + 1))
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        b = df.loc[df['Mã'] == symbol_to_lookup, 'Mã'].values[0]

        start_year = 2018
        end_year = 2022

        selected_years = range(start_year, end_year + 1)
        gross_profit_margins = []
        net_profit_margins = []
        operating_margins = []
        ROAs = []
        ROEs = []
        ROICs = []
        BEPs = []
        EMs = []
        ROEEs = []
        for year in selected_years:
            try:
                Gross_profit = \
                df.loc[df['Mã'] == b, f'KQKD. Lợi nhuận gộp về bán hàng và cung cấp dịch vụ {year}'].values[0]
                Sales = df.loc[df['Mã'] == b, f'KQKD. Doanh thu bán hàng và cung cấp dịch vụ {year}'].values[0]
                gross_profit_margin = ty_le_lai_gop(Gross_profit, Sales)
                gross_profit_margins.append(gross_profit_margin)

                Net_income = df.loc[df['Mã'] == b, f'KQKD. Doanh thu thuần {year}'].values[0]
                net_profit_margin = ty_le_lai_rong(Net_income, Sales)
                net_profit_margins.append(net_profit_margin)

                EBIT_value = EBIT()
                operating_margin = ty_le_loi_nhuan_hoat_dong(EBIT_value, Sales)
                operating_margins.append(operating_margin)

                TA = df.loc[df['Mã'] == b, f'CĐKT. TỔNG CỘNG TÀI SẢN {year}'].values[0]
                ROA_value = ROA(Net_income, TA)
                ROAs.append(ROA_value)

                CE = df.loc[df['Mã'] == b, f'CĐKT. VỐN CHỦ SỞ HỮU {year}'].values[0]
                ROE_value = ROE(Net_income, CE)
                ROEs.append(ROE_value)

                T = 0.2
                Debt = df.loc[df['Mã'] == b, f'CĐKT. NỢ PHẢI TRẢ {year}'].values[0]
                Equity = df.loc[df['Mã'] == b, f'CĐKT. VỐN CHỦ SỞ HỮU {year}'].values[0]
                ROIC_value = ROIC(EBIT_value, T, Debt, Equity)
                ROICs.append(ROIC_value)

                BEP_value = BEP(EBIT_value, TA)
                BEPs.append(BEP_value)

                EM_value = he_so_nhan(TA, Equity)
                EMs.append(EM_value)

                ROEE_value = DUPONT_EQUATION(net_profit_margin, BEP_value, EM_value)
                ROEEs.append(ROEE_value)

            except IndexError:
                print(f"Giá trị {b} không tồn tại trong cột 'Mã'.")
        data = {'Năm': selected_years, 'Gross profit margin': gross_profit_margins,
                'Net profit margin': net_profit_margins, 'Operating margin': operating_margins}
        data1 = {'Năm': selected_years, 'ROA value': ROAs, 'ROE value': ROEs}
        data2 = {'Năm': selected_years, 'ROIC value': ROICs, 'Basic earning power value': BEPs,
                 'Equity Multiplier value': EMs}
        data3 = {'Năm': selected_years, 'DUPONT': ROEEs}
        datatable = [data, data1, data2, data3]

        # In bảng hiển thị kết quả
        for data_dict in datatable:
            df_table = pd.DataFrame(data_dict)
            print(df_table)

        # Chỉ số và tên chỉ số
        indices = ['ROA', 'ROE', 'ROIC', 'BEP', 'EM', 'ROEE', 'Operating Margins']
        values = [ROAs, ROEs, ROICs, BEPs, EMs, ROEEs, operating_margins]

        # Tạo mảng màu sắc cho các cột chồng
        colors = ['cornflowerblue', 'lightcoral', 'mediumseagreen', 'gold', 'slateblue', 'darkorange', 'orchid']

        # Tạo biểu đồ cột chồng
        plt.figure(figsize=(10, 6))
        bottom = np.zeros(len(selected_years))

        for i in range(len(indices)):
            plt.bar(selected_years, values[i], bottom=bottom, color=colors[i], label=indices[i])
            bottom += values[i]

        plt.xlabel('Năm')
        plt.ylabel('Giá trị')
        plt.title(f'Các chỉ số cho {b}')
        plt.legend()
        plt.xticks(range(start_year, end_year + 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()






#Tỷ số thanh khoản
def tinh_ty_so_thanh_toan_hien_hanh(CA, TA):
    CR = CA / TA
    return CR
def tinh_ty_so_thanh_toan_nhanh(CA,In,TA):
    QR = (CA - In) / TA
    return QR
#Tỷ số quản lý tài sản
def vong_quay_hang_ton_kho(Sales,In):
    In_TO =  Sales / In
    return In_TO
def ky_thu_tien_binh_quan(AR,Sales):
    DSO = AR / (Sales/365)
    return DSO
def vong_quay_TSCD(Sales,Net_fixed_assets):
    FA_turnover = Sales / Net_fixed_assets
    return FA_turnover
def vong_quay_TTS(Sales,TA):
    TA_turnover = Sales / TA
    return TA_turnover
#Tỷ số quản lý nợ
def ty_so_no_vay(Total_debt, TA):
    Debt_ratio = Total_debt / TA
    return Debt_ratio


def EBIT(b,year):
    EBIT_value = df.loc[df['Mã'] == b, f"KQKD. Tổng lợi nhuận kế toán trước thuế {year}"].values[0] + df.loc[df['Mã'] == b, f"KQKD. Trong đó: Chi phí lãi vay {year}"].values[0]
    return EBIT_value

def EBITDA(b,year):
    EBIT_value = EBIT()
    EBITDA_value = EBIT_value + df.loc[df['Mã'] == b, f"LCTT. Khấu hao TSCĐ   {year}"].values[0]
    return EBITDA_value

def he_so_chi_tra_lai(EBIT, IE):
    TIE = EBIT / IE
    return TIE

def he_so_kha_nang_tra_no(EBITDA, tttienthue, laivay, no_goc):
    EBITDA_coverage = (EBITDA + tttienthue) / (laivay + tttienthue + no_goc)
    return EBITDA_coverage
# Tỷ số sinh lợi
def ty_le_lai_gop(Gross_profit, Sales):
    Gross_profit_margin = Gross_profit / Sales
    return Gross_profit_margin

def ty_le_lai_rong(Net_income, Sales):
    Net_profit_margin = Net_income / Sales
    return Net_profit_margin

def ty_le_loi_nhuan_hoat_dong(EBIT, Sales):
    Operating_margin = EBIT / Sales
    return Operating_margin

def ROA(Net_income, TA):
    ROA = Net_income / TA
    return ROA

def ROE(Net_income, CE):
    ROE = Net_income / CE
    return ROE

def ROIC(EBIT, T, Debt, Equity):
    ROIC = (EBIT * (1 - T)) / (Debt + Equity)
    return ROIC

def BEP(EBIT, TA):
    BEP = EBIT / TA
    return BEP

def he_so_nhan(TA, Equity):
    EM = TA / Equity
    return EM

def DUPONT_EQUATION(Net_profit_margin, TA_turnover, EM):
    ROEE = Net_profit_margin * TA_turnover * EM
    return ROEE

if __name__ == "__main__":
    data_market=df
    data_price=dfb