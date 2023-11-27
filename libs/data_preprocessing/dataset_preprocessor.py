import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder


class LevelDatasetPreprocessor:
    label_encoder: LabelEncoder = LabelEncoder()
    feature_column_names: List[str] = ["message", "message_detail", "hostname"]
    label_column_name: str = "level"
    label_names: List[str] = ["A", "C", "B", "X", "T", "N"]

    @staticmethod
    def process(processed_df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
        # Select columns
        features_df: pd.DataFrame = processed_df[
            LevelDatasetPreprocessor.feature_column_names
        ]
        label_df: pd.DataFrame = processed_df[
            LevelDatasetPreprocessor.label_column_name
        ]

        LevelDatasetPreprocessor.label_encoder = (
            LevelDatasetPreprocessor.label_encoder.fit(
                LevelDatasetPreprocessor.label_names
            )
        )
        # Integer encoding
        label_df: pd.DataFrame = pd.DataFrame(
            LevelDatasetPreprocessor.label_encoder.transform(label_df)
        )

        return features_df, label_df

    @staticmethod
    def inverse_process(processed_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            LevelDatasetPreprocessor.label_encoder.inverse_transform(processed_df)
        )


class ReasonDetailDatasetPreprocessor:
    label_encoder: LabelEncoder = LabelEncoder()
    feature_column_names: List[str] = ["message", "message_detail", "hostname"]
    label_column_name: str = "reason_detail"
    label_names: List[str] = [
        "時戳服務無法連線，需要立刻通知時戳服務維運團隊",
        "可能為保險公司上傳的檔案異常，或連線到公會網路延遲",
        "可能為網路延遲，或公會端服務異常",
        "可能為保險公司上傳的檔案異常，導致無法簽章",
        "民眾帳號密碼輸入錯誤或尚未註冊",
        "callback不通，可能為保險公司服務維護，或網路不通。待上班時端聯繫保險公司確認。",
        "民眾登入時 OTP 輸入錯誤或已過期",
        "就有資訊，更版後理應不再發生，如有發生需通知工程師確認問題！",
        "保險存摺-投保紀錄查詢新通報資料，資料比對不到",
        "民眾自己密碼輸入錯誤所導致",
        "進行醫起通案件時，公司端呼叫公會失敗，待平日上班時間處理",
        "使用者執行更改密碼時，輸入三代密碼所導致",
        "忘記密碼，民眾輸入的信箱錯誤",
        "檔案搬移失敗，可能原因為硬碟容量問題，待上班時間處理",
        "紙本註記時資料庫查無資料",
        "post到綠界開發票connection time out",
        "民眾修改密碼,密碼輸入錯誤",
        "公司端資料庫無案件",
        "保險存摺至fido server查詢推播資料連線異常",
        "系統更版重新啟動瞬斷",
        "民眾信箱或密碼或著查無帳號輸入錯誤所導致",
        "呼叫API-416失敗，可能原因為API-101參數與API-416不吻合，待上班時間聯繫保險公司",
        "民眾TOKEN已過期，然後返回首頁時發生的錯誤",
        "可移除，無用的log",
        "民眾APP版本太低",
        "民眾申請投保紀錄查詢，案件正在審核中又重複申請所導致",
        "民眾登入過程於個資同意頁連點同意按鈕導致",
        "民眾註冊時身分驗證失敗",
        "進行存證驗證時，未在區塊鏈上查到資料，可能是區塊鏈尚未同步完成，若超過一小時還未完成，需聯繫鄭玉玲。",
        "大部分都是綠界服務打不通所導致",
        "參數設定錯誤導致此問題",
        "重複呼叫公會端",
        "設定錯誤，需調整",
        "理賠sch發生NPT，待上班時上系統確認實際狀況",
        "忘記密碼，民眾輸入的手機錯誤",
    ]

    @staticmethod
    def process(processed_df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
        processed_df[
            processed_df[ReasonDetailDatasetPreprocessor.label_column_name]
            == "檔案搬移失敗,可能原因為硬碟容量問題,待上班時間處理"
        ] = "檔案搬移失敗，可能原因為硬碟容量問題，待上班時間處理"
        processed_df[
            processed_df[ReasonDetailDatasetPreprocessor.label_column_name]
            == "可移除,無用的log"
        ] = "可移除，無用的log"
        processed_df[
            processed_df[ReasonDetailDatasetPreprocessor.label_column_name]
            == "進行存證驗證時,未在區塊鏈上查到資料,可能是區塊鏈尚未同步完成,若超過一小時還未完成,需聯繫鄭玉玲。"
        ] = "進行存證驗證時，未在區塊鏈上查到資料，可能是區塊鏈尚未同步完成，若超過一小時還未完成，需聯繫鄭玉玲。"
        processed_df[
            processed_df[ReasonDetailDatasetPreprocessor.label_column_name]
            == "callback不通,可能為保險公司服務維護,或網路不通。待上班時端聯繫保險公司確認。"
        ] = "callback不通，可能為保險公司服務維護，或網路不通。待上班時端聯繫保險公司確認。"
        processed_df[
            processed_df[ReasonDetailDatasetPreprocessor.label_column_name]
            == "進行醫起通案件時,公司端呼叫公會失敗"
        ] = "進行醫起通案件時，公司端呼叫公會失敗，待平日上班時間處理"
        processed_df[
            processed_df[ReasonDetailDatasetPreprocessor.label_column_name]
            == "設定錯誤,需調整"
        ] = "設定錯誤，需調整"
        processed_df[
            processed_df[ReasonDetailDatasetPreprocessor.label_column_name]
            == "理賠sch發生NPT,待上班時上系統確認實際狀況"
        ] = "理賠sch發生NPT，待上班時上系統確認實際狀況"
        # Select columns
        features_df: pd.DataFrame = processed_df[
            ReasonDetailDatasetPreprocessor.feature_column_names
        ]
        label_df: pd.DataFrame = processed_df[
            ReasonDetailDatasetPreprocessor.label_column_name
        ]

        ReasonDetailDatasetPreprocessor.label_encoder = (
            ReasonDetailDatasetPreprocessor.label_encoder.fit(
                ReasonDetailDatasetPreprocessor.label_names
            )
        )

        # Integer encoding
        label_df: pd.DataFrame = pd.DataFrame(
            ReasonDetailDatasetPreprocessor.label_encoder.transform(label_df)
        )

        return features_df, label_df

    @staticmethod
    def inverse_process(processed_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            ReasonDetailDatasetPreprocessor.label_encoder.inverse_transform(
                processed_df
            )
        )


class SystemDatasetPreprocessor:
    label_encoder: LabelEncoder = LabelEncoder()
    feature_column_names: List[str] = ["message", "message_detail", "hostname"]
    label_column_name: str = "system"
    label_names: List[str] = ["電子保單", "保險存摺", "保全", "身份驗證/文件簽署", "醫起通", "理賠"]

    @staticmethod
    def process(processed_df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
        processed_df[pd.isna(processed_df["system"])] = "電子保單"
        processed_df[
            processed_df[SystemDatasetPreprocessor.label_column_name]
            == "檔案搬移失敗，可能原因為硬碟容量問題，待上班時間處理"
        ] = "電子保單"
        processed_df[
            processed_df[SystemDatasetPreprocessor.label_column_name] == "可移除，無用的log"
        ] = "電子保單"
        processed_df[
            processed_df[SystemDatasetPreprocessor.label_column_name]
            == "callback不通，可能為保險公司服務維護，或網路不通。待上班時端聯繫保險公司確認。"
        ] = "電子保單"
        processed_df[
            processed_df[SystemDatasetPreprocessor.label_column_name]
            == "進行存證驗證時，未在區塊鏈上查到資料，可能是區塊鏈尚未同步完成，若超過一小時還未完成，需聯繫鄭玉玲。"
        ] = "電子保單"
        processed_df[
            processed_df[SystemDatasetPreprocessor.label_column_name]
            == "進行醫起通案件時，公司端呼叫公會失敗，待平日上班時間處理"
        ] = "電子保單"
        processed_df[
            processed_df[SystemDatasetPreprocessor.label_column_name]
            == "理賠sch發生NPT，待上班時上系統確認實際狀況"
        ] = "電子保單"
        processed_df[
            processed_df[SystemDatasetPreprocessor.label_column_name] == "設定錯誤，需調整"
        ] = "電子保單"
        # Select columns
        features_df: pd.DataFrame = processed_df[
            SystemDatasetPreprocessor.feature_column_names
        ]
        label_df: pd.DataFrame = processed_df[
            SystemDatasetPreprocessor.label_column_name
        ]

        SystemDatasetPreprocessor.label_encoder = (
            SystemDatasetPreprocessor.label_encoder.fit(
                SystemDatasetPreprocessor.label_names
            )
        )

        # Integer encoding
        label_df: pd.DataFrame = pd.DataFrame(
            SystemDatasetPreprocessor.label_encoder.transform(label_df)
        )

        return features_df, label_df

    @staticmethod
    def inverse_process(processed_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            SystemDatasetPreprocessor.label_encoder.inverse_transform(processed_df)
        )


class ReasonDatasetPreprocessor:
    label_encoder: LabelEncoder = LabelEncoder()
    feature_column_names: List[str] = ["message", "message_detail", "hostname"]
    label_column_name: str = "reason"
    label_names: List[str] = ["網路,系統服務", "網路,保險公司", "網路", "保險公司", "使用者行為", "系統服務"]

    @staticmethod
    def process(processed_df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
        processed_df[
            processed_df[ReasonDatasetPreprocessor.label_column_name] == "電子保單"
        ] = "網路,保險公司"
        # Select columns
        features_df: pd.DataFrame = processed_df[
            ReasonDatasetPreprocessor.feature_column_names
        ]
        label_df: pd.DataFrame = processed_df[
            ReasonDatasetPreprocessor.label_column_name
        ]

        ReasonDatasetPreprocessor.label_encoder = (
            ReasonDatasetPreprocessor.label_encoder.fit(
                ReasonDatasetPreprocessor.label_names
            )
        )
        # Integer encoding
        label_df: pd.DataFrame = pd.DataFrame(
            ReasonDatasetPreprocessor.label_encoder.transform(label_df)
        )

        return features_df, label_df

    @staticmethod
    def inverse_process(processed_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            ReasonDatasetPreprocessor.label_encoder.inverse_transform(processed_df)
        )


class PersonDatasetPreprocessor:
    label_encoder: LabelEncoder = LabelEncoder()
    feature_column_names: List[str] = ["message", "message_detail", "hostname"]
    label_column_name: str = "person"
    label_names: List[str] = ["張天瑋", "吳健平", "蔡逸丞", "張家璽", "陳胤中"]

    @staticmethod
    def process(processed_df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
        processed_df[
            processed_df[PersonDatasetPreprocessor.label_column_name] == "網路,保險公司"
        ] = "張家璽"
        # Select columns
        features_df: pd.DataFrame = processed_df[
            PersonDatasetPreprocessor.feature_column_names
        ]
        label_df: pd.DataFrame = processed_df[
            PersonDatasetPreprocessor.label_column_name
        ]

        PersonDatasetPreprocessor.label_encoder = (
            PersonDatasetPreprocessor.label_encoder.fit(
                PersonDatasetPreprocessor.label_names
            )
        )
        # Integer encoding
        label_df: pd.DataFrame = pd.DataFrame(
            PersonDatasetPreprocessor.label_encoder.transform(label_df)
        )

        return features_df, label_df

    @staticmethod
    def inverse_process(processed_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            PersonDatasetPreprocessor.label_encoder.inverse_transform(processed_df)
        )


def main():
    pass


if __name__ == "__main__":
    main()
