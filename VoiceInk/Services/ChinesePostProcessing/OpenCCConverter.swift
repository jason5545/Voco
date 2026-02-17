import Foundation

/// OpenCC s2twp converter wrapper
/// Converts Simplified Chinese to Traditional Chinese (Taiwan) with phrase conversion
/// Reference: xvoice/src/chinese.py
class OpenCCConverter {
    static let shared = OpenCCConverter()

    private let converter: Any? // SwiftyOpenCC.ChineseConverter when available
    private let convertFunction: ((String) -> String)?

    private init() {
        // Attempt to load SwiftyOpenCC at runtime
        // If SwiftyOpenCC is not available, fall back to a basic conversion table
        do {
            let conv = try Self.createConverter()
            self.converter = conv.converter
            self.convertFunction = conv.convert
        } catch {
            self.converter = nil
            self.convertFunction = nil
        }
    }

    private static func createConverter() throws -> (converter: Any, convert: (String) -> String) {
        // This will be linked at compile time when SwiftyOpenCC is added as SPM dependency
        // For now, provide a stub that uses the built-in fallback table
        throw ConverterError.notAvailable
    }

    private enum ConverterError: Error {
        case notAvailable
    }

    /// Convert Simplified Chinese to Traditional Chinese (Taiwan with phrases)
    func convert(_ text: String) -> String {
        if let convertFunction = convertFunction {
            return convertFunction(text)
        }
        // Fallback: basic s2tw conversion table for common characters
        return fallbackConvert(text)
    }

    // MARK: - Fallback conversion (subset of s2twp)

    /// Basic fallback conversion for the most common simplified-to-traditional mappings
    /// encountered in speech recognition output
    private func fallbackConvert(_ text: String) -> String {
        var result = text

        // Phrase-level conversions (s2twp specific - Taiwan phrases)
        for (simplified, traditional) in Self.phraseMappings {
            result = result.replacingOccurrences(of: simplified, with: traditional)
        }

        // Character-level conversions
        var chars = Array(result)
        for i in 0..<chars.count {
            if let traditional = Self.charMappings[chars[i]] {
                chars[i] = traditional
            }
        }

        return String(chars)
    }

    /// Common phrase mappings (Taiwan-specific, s2twp mode)
    private static let phraseMappings: [(String, String)] = [
        // Software/tech terms
        ("软件", "軟體"),
        ("硬件", "硬體"),
        ("程序", "程式"),
        ("内存", "記憶體"),
        ("存储", "儲存"),
        ("默认", "預設"),
        ("信息", "資訊"),
        ("视频", "影片"),
        ("音频", "音訊"),
        ("网络", "網路"),
        ("服务器", "伺服器"),
        ("数据库", "資料庫"),
        ("数据", "資料"),
        ("文件夹", "資料夾"),
        ("文档", "文件"),
        ("打印", "列印"),
        ("鼠标", "滑鼠"),
        ("光标", "游標"),
        ("移动", "行動"),
        ("U盘", "隨身碟"),
        ("优盘", "隨身碟"),
        ("激光", "雷射"),
        ("博客", "部落格"),
        ("短信", "簡訊"),
        ("字节", "位元組"),
        ("比特", "位元"),
        ("宽带", "寬頻"),
        ("高清", "高畫質"),
        ("智能", "智慧"),
        ("人工智能", "人工智慧"),
        ("机器人", "機器人"),
    ]

    /// Common character-level simplified to traditional mappings
    private static let charMappings: [Character: Character] = [
        "与": "與", "专": "專", "业": "業", "东": "東", "两": "兩",
        "个": "個", "丰": "豐", "临": "臨", "为": "為", "举": "舉",
        "义": "義", "乐": "樂", "书": "書", "买": "買", "乱": "亂",
        "争": "爭", "于": "於", "亏": "虧", "云": "雲", "产": "產",
        "亲": "親", "仅": "僅", "从": "從", "仓": "倉", "会": "會",
        "伟": "偉", "传": "傳", "伤": "傷", "众": "眾", "优": "優",
        "价": "價", "华": "華", "协": "協", "单": "單", "卖": "賣",
        "历": "歷", "厂": "廠", "发": "發", "变": "變", "号": "號",
        "叶": "葉", "听": "聽", "员": "員", "响": "響", "图": "圖",
        "团": "團", "国": "國", "园": "園", "场": "場", "坏": "壞",
        "块": "塊", "处": "處", "备": "備", "头": "頭", "夸": "誇",
        "实": "實", "对": "對", "导": "導", "将": "將", "尔": "爾",
        "层": "層", "岁": "歲", "属": "屬", "带": "帶", "师": "師",
        "帮": "幫", "广": "廣", "应": "應", "开": "開", "异": "異",
        "张": "張", "归": "歸", "录": "錄", "总": "總", "惊": "驚",
        "战": "戰", "户": "戶", "执": "執", "护": "護", "报": "報",
        "择": "擇", "挡": "擋", "损": "損", "换": "換", "据": "據",
        "摄": "攝", "断": "斷", "无": "無", "旧": "舊", "时": "時",
        "显": "顯", "术": "術", "机": "機", "权": "權", "条": "條",
        "来": "來", "标": "標", "样": "樣", "检": "檢", "楼": "樓",
        "欢": "歡", "残": "殘", "气": "氣", "汇": "匯", "没": "沒",
        "况": "況", "济": "濟", "测": "測", "浏": "瀏", "点": "點",
        "热": "熱", "爱": "愛", "环": "環", "现": "現", "电": "電",
        "画": "畫", "种": "種", "称": "稱", "积": "積", "笔": "筆",
        "类": "類", "紧": "緊", "纪": "紀", "约": "約", "级": "級",
        "纯": "純", "线": "線", "组": "組", "经": "經", "结": "結",
        "绝": "絕", "统": "統", "继": "繼", "绩": "績", "续": "續",
        "网": "網", "联": "聯", "节": "節", "范": "範", "营": "營",
        "虑": "慮", "虽": "雖", "补": "補", "观": "觀", "规": "規",
        "览": "覽", "认": "認", "让": "讓", "论": "論", "设": "設",
        "证": "證", "评": "評", "识": "識", "话": "話", "语": "語",
        "说": "說", "请": "請", "调": "調", "谁": "誰", "负": "負",
        "质": "質", "费": "費", "资": "資", "赶": "趕", "转": "轉",
        "载": "載", "运": "運", "进": "進", "选": "選", "递": "遞",
        "通": "通", "连": "連", "远": "遠", "适": "適", "还": "還",
        "这": "這", "里": "裡", "量": "量", "长": "長", "门": "門",
        "间": "間", "关": "關", "阅": "閱", "队": "隊", "际": "際",
        "险": "險", "随": "隨", "难": "難", "集": "集", "须": "須",
        "顾": "顧", "飞": "飛", "验": "驗", "鸟": "鳥",
    ]
}
