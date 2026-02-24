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
        // Fallback: phrase + char table first, then Apple's built-in S→T as safety net
        let result = fallbackConvert(text)
        return appleSimplifiedToTraditional(result)
    }

    /// Use Apple's built-in CFStringTransform to catch any remaining simplified characters
    /// that the fallback table missed.
    private func appleSimplifiedToTraditional(_ text: String) -> String {
        let mutable = NSMutableString(string: text)
        CFStringTransform(mutable, nil, "Simplified-Traditional" as CFString, false)
        return mutable as String
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
    /// Phrase mappings run BEFORE character mappings so ambiguous characters
    /// (只/制/系/干/后/台 etc.) are resolved in context first.
    private static let phraseMappings: [(String, String)] = [
        // Software/tech terms
        ("软件", "軟體"),
        ("软体", "軟體"),
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
        ("搜索", "搜尋"),
        ("短信", "簡訊"),
        ("字节", "位元組"),
        ("比特", "位元"),
        ("宽带", "寬頻"),
        ("高清", "高畫質"),
        ("智能", "智慧"),
        ("人工智能", "人工智慧"),
        ("机器人", "機器人"),

        // Ambiguous char: 么 — phrase-level to avoid edge cases
        ("什么", "什麼"),
        ("怎么", "怎麼"),
        ("那么", "那麼"),
        ("多么", "多麼"),
        ("要么", "要麼"),
        ("么样", "麼樣"),

        // Ambiguous char: 只 — 只有/只是 stays 只, measure word → 隻
        ("一只", "一隻"),
        ("两只", "兩隻"),
        ("几只", "幾隻"),
        ("那只", "那隻"),
        ("这只", "這隻"),
        ("每只", "每隻"),

        // Ambiguous char: 制 — 制度 stays 制, manufacturing → 製
        ("制造", "製造"),
        ("制作", "製作"),
        ("制品", "製品"),
        ("录制", "錄製"),
        ("拍制", "拍製"),

        // Ambiguous char: 系 — 系統 stays 系, relation → 係
        ("关系", "關係"),
        ("没关系", "沒關係"),

        // Ambiguous char: 干 — context dependent
        ("干净", "乾淨"),
        ("干燥", "乾燥"),
        ("饼干", "餅乾"),
        ("干脆", "乾脆"),
        ("干嘛", "幹嘛"),
        ("干什", "幹什"),
        ("干吗", "幹嗎"),

        // Ambiguous char: 后 — 后面/以后 → 後, 皇后 stays 后
        ("以后", "以後"),
        ("之后", "之後"),
        ("后来", "後來"),
        ("后面", "後面"),
        ("然后", "然後"),
        ("后天", "後天"),
        ("最后", "最後"),
        ("前后", "前後"),
        ("后果", "後果"),
        ("后续", "後續"),
        ("后悔", "後悔"),
        ("落后", "落後"),
        ("后方", "後方"),
        ("后边", "後邊"),
        ("后门", "後門"),
        ("背后", "背後"),
        ("后期", "後期"),
        ("后半", "後半"),
        ("后排", "後排"),
        ("后座", "後座"),

        // Ambiguous char: 台 — 台灣 stays 台, 舞台/台上 context
        ("台风", "颱風"),

        // Common speech phrases
        ("标准", "標準"),
        ("准确", "準確"),
        ("准备", "準備"),
        ("批准", "批准"),
        ("获得", "獲得"),
        ("仿佛", "彷彿"),
    ]

    /// Common character-level simplified to traditional mappings
    /// Note: ambiguous chars (只/制/系/干/后/台/准/么) are handled by phraseMappings above
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
        // [AI-Claude: 2026-02-18] Added missing chars found in confidence-routing.log
        "吗": "嗎", "问": "問", "题": "題", "讲": "講", "该": "該",
        "软": "軟", "达": "達", "够": "夠", "厉": "厲", "状": "狀",
        "脚": "腳", "体": "體", "养": "養", "写": "寫", "别": "別",
        // Additional common speech recognition simplified chars
        "获": "獲", "给": "給", "刚": "剛", "医": "醫", "办": "辦",
        "能": "能", "岛": "島", "尽": "盡", "虫": "蟲", "牺": "犧",
        "决": "決", "兴": "興", "杂": "雜", "详": "詳", "庆": "慶",
        "弃": "棄", "贴": "貼", "铁": "鐵", "钱": "錢", "钟": "鐘",
        "银": "銀", "锁": "鎖", "键": "鍵", "镜": "鏡", "闻": "聞",
        "阳": "陽", "阴": "陰", "陆": "陸", "陈": "陳", "隐": "隱",
        "饭": "飯", "饮": "飲", "怀": "懷", "忆": "憶", "惯": "慣",
        "愿": "願", "态": "態", "忧": "憂", "拥": "擁", "担": "擔",
        "抢": "搶", "挤": "擠", "挥": "揮", "拟": "擬", "搜": "搜",
        "压": "壓", "坚": "堅", "垫": "墊", "壮": "壯", "声": "聲",
        "壳": "殼", "宝": "寶", "宁": "寧", "宫": "宮",
        "宽": "寬", "审": "審", "宪": "憲", "窝": "窩", "窍": "竅",
        "竞": "競", "笼": "籠", "签": "簽", "简": "簡", "粮": "糧",
        "纲": "綱", "纳": "納", "纵": "縱", "纷": "紛", "纸": "紙",
        "细": "細", "织": "織", "终": "終", "绍": "紹", "绕": "繞",
        "绘": "繪", "络": "絡", "绿": "綠", "缘": "緣",
        "缩": "縮", "编": "編", "缺": "缺", "罗": "羅", "罚": "罰",
        "药": "藥", "荣": "榮", "蓝": "藍", "藏": "藏", "蚁": "蟻",
        "蛇": "蛇", "衬": "襯", "袜": "襪", "裤": "褲", "误": "誤",
        "读": "讀", "课": "課", "谈": "談", "谊": "誼", "谢": "謝",
        "谱": "譜", "贝": "貝", "贡": "貢", "贫": "貧", "购": "購",
        "贯": "貫", "赏": "賞", "赔": "賠", "赢": "贏", "赵": "趙",
        "跃": "躍", "践": "踐", "踪": "蹤", "轮": "輪", "轰": "轟",
        "辅": "輔", "辆": "輛", "辈": "輩", "辉": "輝", "边": "邊",
        "过": "過", "迁": "遷", "迈": "邁", "违": "違",
        "遥": "遙", "邮": "郵", "郁": "鬱", "针": "針", "钻": "鑽",
        "铃": "鈴", "铅": "鉛", "链": "鏈", "锅": "鍋", "锋": "鋒",
        "锻": "鍛", "镇": "鎮", "闪": "閃", "阀": "閥", "阁": "閣",
        "阵": "陣", "阶": "階", "障": "障", "雾": "霧", "零": "零",
        "韩": "韓", "顿": "頓", "颗": "顆", "频": "頻", "颜": "顏",
        "风": "風", "饱": "飽", "馆": "館", "驱": "驅", "骗": "騙",
        "鱼": "魚", "鸡": "雞", "麦": "麥", "龙": "龍", "龄": "齡",
    ]
}
