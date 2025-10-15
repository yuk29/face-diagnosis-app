import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import time
import pandas as pd



# MediaPipe設定
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class AdvancedFaceTypeAnalyzer:
    """大人顔・子供顔判別と曲線・直線要素分析による高精度顔タイプ診断システム"""
    
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 顔の主要ランドマークインデックス
        self.landmark_indices = {
            'facehead_top': 10,
            'forehead_left': 103,
            'forehead_right': 332,
            'eyebrow_left_inner': 70,
            'eyebrow_left_outer': 46,
            'eyebrow_right_inner': 300,
            'eyebrow_right_outer': 276,
            'brow_center': 9,
            'eye_left_inner': 133,
            'eye_left_outer': 33,
            'eye_right_inner': 362,
            'eye_right_outer': 263,
            'eye_left_top': 159,
            'eye_left_bottom': 145,
            'eye_right_top': 386,
            'eye_right_bottom': 374,
            'nose_tip': 1,
            'nose_root':168,
            'nose_bridge': 6,
            'nose_left': 131,
            'nose_right': 360,
            'nose_bottom': 164,
            'mouth_left': 61,
            'mouth_right': 291,
            'mouth_top': 0,
            'mouth_bottom': 17,
            'chin': 152,
            'jaw_left': 172,
            'jaw_right': 397,
            'left_cheekbone': 127,
            'right_cheekbone': 356,
            'cheek_left': 116,
            'cheek_right': 345,
            'upper_lip': 12,
            'lower_lip': 15
        }
    
    def get_distance(self, p1, p2):
        """2点間の距離を計算"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def get_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """3点から成す角度を計算"""
        a = self.get_distance(p2, p3)
        b = self.get_distance(p1, p3)
        c = self.get_distance(p1, p2)
        
        if a == 0 or c == 0:
            return 0
        
        cos_angle = (a**2 + c**2 - b**2) / (2 * a * c)
        cos_angle = max(-1, min(1, cos_angle))
        return math.degrees(math.acos(cos_angle))
    
    def analyze_adult_child_features(self, landmarks) -> Tuple[int, Dict]:
        """大人顔・子供顔の特徴を分析"""
        coords = {k: (landmarks[v].x, landmarks[v].y) for k, v in self.landmark_indices.items()}
        
        adult_score = 0
        child_score =0
        analysis = {}
        
        # 1. 顔の縦横比（縦が短い=子供、長い=大人）
        face_height = self.get_distance(coords['facehead_top'], coords['chin'])
        face_width = self.get_distance(coords['left_cheekbone'], coords['right_cheekbone'])
        face_ratio = face_width / face_height if face_height > 0 else 0
        
        if face_ratio < 0.91:  # 縦長
            adult_score += 1
            analysis['face_ratio'] = f'大人要素（縦長）- 顔比: {face_ratio:.3f} < 0.91'
        else:
            child_score += 1
            analysis['face_ratio'] = f'子供要素（横幅広め）- 顔比: {face_ratio:.3f} ≥ 0.91'
        
        # 2. あごの長さ（短い=子供、長い=大人）
        mouth_to_chin = self.get_distance(coords['mouth_bottom'], coords['chin'])
        face_under = self.get_distance(coords['nose_bottom'], coords['chin'])
        chin_ratio = mouth_to_chin / face_under if face_height > 0 else 0
        
        if chin_ratio > 0.51:  # あご長め
            adult_score += 1
            analysis['chin_length'] = f'大人要素（あご長め） - あご比: {chin_ratio:.3f} > 0.54'
        else:
            child_score += 1
            analysis['chin_length'] = f'子供要素（あご短め） - あご比: {chin_ratio:.3f} ≤ 0.54'
        
        # 3. 目の間隔（離れ気味=子供、寄り気味=大人）
        eye_distance = self.get_distance(coords['eye_left_inner'], coords['eye_right_inner'])
        eye_distance_ratio = eye_distance / face_width if face_width > 0 else 0
        
        if eye_distance_ratio < 0.255:  # 目が寄り気味
            adult_score += 1
            analysis['eye_distance'] = f'大人要素（目が寄り気味） - 目間隔比: {eye_distance_ratio:.3f} < 0.25'
        else:
            child_score += 1
            analysis['eye_distance'] = f'子供要素（目が離れ気味） - 目間隔比: {eye_distance_ratio:.3f} ≥ 0.25'
        
        # 4. 目の大きさ（小さい=大人、大きい=子供）
        eye_left_width = self.get_distance(coords['eye_left_inner'], coords['eye_left_outer'])
        eye_right_width = self.get_distance(coords['eye_right_inner'], coords['eye_right_outer'])
        avg_eye_width = (eye_left_width + eye_right_width) / 2
        eye_size_ratio = avg_eye_width / face_width if face_width > 0 else 0
        
        if eye_size_ratio > 0.199:  # 目が大きめ
            adult_score += 1
            analysis['eye_size'] = f'大人要素（目が大きめ） - 目サイズ比: {eye_size_ratio:.3f} > 0.19'
        else:
            child_score += 1
            analysis['eye_size'] = f'子供要素（目が小さめ） - 目サイズ比: {eye_size_ratio:.3f} ≤ 0.19'
        
        # 5. 小鼻の横幅（目1つ分より小さい=子供、大きい=大人）
        nose_width = self.get_distance(coords['nose_left'], coords['nose_right'])
        nose_to_eye_ratio = nose_width / avg_eye_width if avg_eye_width > 0 else 0
        
        if nose_to_eye_ratio > 1.1:  # 鼻幅が目幅より大きい
            adult_score += 1
            analysis['nose_width'] = f'大人要素（鼻幅広） - 鼻/目比: {nose_to_eye_ratio:.3f} > 1.1'
        else:
            child_score += 1
            analysis['nose_width'] = f'子供要素（鼻幅狭） - 鼻/目比: {nose_to_eye_ratio:.3f} ≤ 1.1'
        
        return adult_score, child_score, analysis
    
    def analyze_curve_straight_features(self, landmarks) -> Tuple[int, Dict]:
        """曲線・直線要素を分析"""
        coords = {k: (landmarks[v].x, landmarks[v].y) for k, v in self.landmark_indices.items()}
        
        curve_score = 0
        straight_score = 0
        analysis = {}

        
        # 1. 目の形（丸く縦幅=曲線、切れ長=直線）
        eye_left_height = self.get_distance(coords['eye_left_top'], coords['eye_left_bottom'])
        eye_left_width = self.get_distance(coords['eye_left_inner'], coords['eye_left_outer'])
        eye_right_height = self.get_distance(coords['eye_right_top'], coords['eye_right_bottom'])
        eye_right_width = self.get_distance(coords['eye_right_inner'], coords['eye_right_outer'])
        eye_aspect_ratio = ( eye_left_height / eye_left_width + eye_right_height / eye_right_width ) / 2 if eye_left_width > 0 else 0

        
        if eye_aspect_ratio > 0.35:  # 縦幅がある丸い目
            curve_score += 1
            analysis['eye_shape'] = f'曲線要素（丸い目） - 目縦横比: {eye_aspect_ratio:.3f} > 0.35'
        else:
            straight_score += 1
            analysis['eye_shape'] = f'直線要素（切れ長） - 目縦横比: {eye_aspect_ratio:.3f} ≤ 0.35'
        
        # 2. 目の傾き（たれ目=曲線、つり目=直線）
        # 左目の傾き（目頭→目尻の傾き）
        eye_left_slope = (coords['eye_left_outer'][1] - coords['eye_left_inner'][1]) / \
                         (coords['eye_left_outer'][0] - coords['eye_left_inner'][0]) \
                         if coords['eye_left_outer'][0] != coords['eye_left_inner'][0] else 0
        # 右目の傾き（目頭→目尻の傾き）
        eye_right_slope = (coords['eye_right_outer'][1] - coords['eye_right_inner'][1]) / \
                          (coords['eye_right_outer'][0] - coords['eye_right_inner'][0]) \
                          if coords['eye_right_outer'][0] != coords['eye_right_inner'][0] else 0
        # 平均傾き
        avg_eye_slope = (eye_left_slope + eye_right_slope) / 2
        # 判定（Yが上に行くほど小さいので、傾きがマイナス＝目尻が上＝つり目）
        if avg_eye_slope < 0:
            straight_score += 1
            analysis['eye_tilt'] = f'直線要素（つり目） - 目傾斜: {avg_eye_slope:.3f} < 0°'
        else:
            curve_score += 1
            analysis['eye_tilt'] = f'曲線要素（たれ目） - 目傾斜: {avg_eye_slope:.3f} ≥ 0°'
        
        # 3. 唇の厚み（厚い=曲線、薄い=直線）
        lip_thickness = self.get_distance(coords['mouth_top'], coords['mouth_bottom'])
        face_under = self.get_distance(coords['nose_bottom'], coords['chin'])
        lip_thickness_ratio = lip_thickness / face_under if face_under > 0 else 0
        
        if lip_thickness_ratio > 0.32:  # 厚い唇
            curve_score += 1
            analysis['lip_thickness'] = f'曲線要素（厚い唇） - 唇厚比: {lip_thickness_ratio:.4f} > 0.32'
        else:
            straight_score += 1
            analysis['lip_thickness'] = f'直線要素（薄い唇） - 唇厚比: {lip_thickness_ratio:.4f} ≤ 0.32'
        
        # 4. 顔の形（丸顔=曲線、面長=直線）
        face_under = self.get_distance(coords['nose_bottom'], coords['chin'])
        middle_height = self.get_distance(coords['brow_center'], coords['nose_bottom'])
        face_ratio = middle_height / face_under if face_under > 0 else 0
        
        if face_ratio < 1.4:  # 丸顔
            curve_score += 1
            analysis['face_shape'] = f'曲線要素（丸顔） - 顔縦比: {face_ratio:.3f} < 1.4'
        else:
            straight_score += 1
            analysis['face_shape'] = f'直線要素（面長） - 顔縦比: {face_ratio:.3f} ≥ 1.4'
        
        # 5. まぶた（二重=曲線、一重または奥二重=直線）
        eyelid_type = st.session_state.get('eyelid_type', '未選択')

        if eyelid_type == "二重":
            curve_score += 1
            analysis['eyelid'] = "曲線要素（二重）"
        else:
            straight_score += 1
            analysis['eyelid'] = "直線要素（一重または奥二重）"

        
        #6.顎の形（角度が大きい=曲線、角度が小さい=直線）
        #jaw_angle = self.get_angle(coords['jaw_left'], coords['chin'], coords['jaw_right'])
        #if jaw_angle < 110:
        #    straight_score += 1
        #    analysis['jaw_sharpness'] = f'直線要素（シャープな顎） - 顎角度: {jaw_angle:.2f}° < 110°'
        #else:
        #    curve_score += 1
        #    analysis['jaw_sharpness'] = f'曲線要素（丸みのある顎） - 顎角度: {jaw_angle:.2f}° ≥ 110°'

        return curve_score, straight_score, analysis
    
    def diagnose_face_type(self, adult_score: int, curve_score: int, straight_score: int, eye_size_ratio: float) -> str:
        """大人顔・子供顔と曲線・直線要素から顔タイプを診断"""
        is_adult = adult_score >= 3  # 5項目中3項目以上で大人顔
        
        if curve_score >= 4:  # 曲線要素が5～6個
            if is_adult:
                return 'フェミニン'
            else:
                # 目の大きさで判別
                if eye_size_ratio > 0.26:
                    return 'アクティブキュート'
                else:
                    return 'キュート'
        elif straight_score >= 4:   # 直線要素が5～6個
            if is_adult:
                return 'クール'
            else:
                return 'クールカジュアル'
        else:  # 中間
            if not is_adult:
                return 'フレッシュ'
            else:  # 大人顔
                # 目の大きさで判別
                if eye_size_ratio > 0.26:
                    return 'エレガント'
                else:
                    return 'ソフトエレガント'
    
    def get_face_type_description(self, face_type: str) -> str:
        """顔タイプの説明を返す"""
        descriptions = {
            'キュート': '丸顔で童顔、可愛らしい印象。目が大きく、全体的に柔らかい印象',
            'アクティブキュート': '横幅広めで丸顔、元気で活発な印象。親しみやすい雰囲気',
            'フェミニン': 'やや丸顔で優しい印象、女性らしい美しさ。上品で温かみがある',
            'ソフトエレガント': '上品で柔らかい印象、優雅な美しさ。バランスの取れた顔立ち',
            'エレガント': '縦長でシャープな印象、洗練された美しさ。知的で上品',
            'クール': 'やや縦長で直線的、知的でクールな印象。シャープな輪郭',
            'フレッシュ': 'バランスの良い輪郭、爽やかで親しみやすい。自然な美しさ',
            'クールカジュアル': 'やや横幅広め、カジュアルで親近感がある。リラックスした印象'
        }
        return descriptions.get(face_type, '特徴を分析中...')

def main():
    st.set_page_config(
        page_title="高精度顔タイプ診断アプリ",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("🎭 高精度顔タイプ診断アプリ")
    st.markdown("**大人顔・子供顔判別 × 曲線・直線要素分析による精密診断**")
    
    # サイドバー設定
    st.sidebar.header("設定")
    show_landmarks = st.sidebar.checkbox("顔の特徴点を表示", False)
    show_detailed_analysis = st.sidebar.checkbox("詳細分析を表示", True)
    show_diagnostic_process = st.sidebar.checkbox("診断プロセスを表示", True)
    
    # 診断方法の説明
    with st.expander("🔍 診断方法について"):
        st.markdown("""
        ### 大人顔・子供顔の判別基準
        1. **顔の縦横比**: 縦が短い=子供顔、長い=大人顔
        2. **あごの長さ**: 短い=子供顔、長い=大人顔  
        3. **目の間隔**: 離れ気味=子供顔、寄り気味=大人顔
        4. **目の大きさ**: 大きい=子供顔、小さい=大人顔
        5. **小鼻の横幅**: 目1つ分より小さい=子供顔、大きい=大人顔
        
        ### 曲線・直線要素の判別基準
        1. **目の形**: 丸く縦幅がある=曲線、切れ長=直線
        2. **目の傾き**: たれ目=曲線、つり目=直線
        3. **唇の厚み**: 厚い=曲線、薄い=直線
        4. **顔の形**: 丸顔=曲線、面長=直線
        5. **頬の形**: 丸み=曲線、えら張り=直線
        6. **眉の形**: 曲線的=曲線、直線的=直線
        
        ### 最終診断ルール
        - **曲線要素5-6個 + 大人顔** → フェミニン
        - **曲線要素5-6個 + 子供顔 + 目大** → アクティブキュート
        - **曲線要素5-6個 + 子供顔 + 目小** → キュート
        - **曲線要素2-4個 + 子供顔** → フレッシュ
        - **曲線要素2-4個 + 大人顔 + 目大** → エレガント
        - **曲線要素2-4個 + 大人顔 + 目小** → ソフトエレガント
        - **直線要素5-6個 + 大人顔** → クール
        - **直線要素5-6個 + 子供顔** → クールカジュアル
        """)
    
        # セッション状態で選択を保持
        if 'eyelid_type' not in st.session_state:
            st.session_state['eyelid_type'] = "未選択"

        # 初期選択
        eyelid_type = st.selectbox(
            "最初に瞼の種類を選択してください：",
            ["未選択", "一重または奥二重", "二重"]
        )

        # 選択結果をセッションに保存
        st.session_state['eyelid_type'] = eyelid_type

        # 「未選択」なら診断スタートさせない
        if st.session_state['eyelid_type'] == "未選択":
            st.warning("診断を始める前に、瞼の種類を選択してください。")
            st.stop()

    # メインエリア
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 リアルタイム診断")
        
        # Webカメラの設定
        if "run" not in st.session_state:     
            st.session_state["run"] = False
        if st.button("▶️ カメラを開始 / 停止"):
            st.session_state["run"] = not st.session_state["run"]

        run = st.session_state["run"] 
        FRAME_WINDOW = st.image([])
        
    with col2:
        st.subheader("📊 診断結果")
        diagnosis_placeholder = st.empty()
        analysis_placeholder = st.empty()
    
    # 顔分析器の初期化
    analyzer = AdvancedFaceTypeAnalyzer()

    # 画像診断セクション
    with st.expander("🖼️ 画像で診断する"):
        uploaded_file = st.file_uploader("画像をアップロード（顔が正面のもの）", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            results = analyzer.face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    if show_landmarks:
                        annotated = image_rgb.copy()
                        mp_drawing.draw_landmarks(
                            image=annotated,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                        st.image(annotated, caption="検出結果", channels="RGB")

                    # 分析実行
                    adult_score, child_score, adult_analysis = analyzer.analyze_adult_child_features(face_landmarks.landmark)
                    curve_score, straight_score, curve_analysis = analyzer.analyze_curve_straight_features(face_landmarks.landmark)

                    # 目の大きさ比率を計算
                    coords = {k: (face_landmarks.landmark[v].x, face_landmarks.landmark[v].y) for k, v in analyzer.landmark_indices.items()}
                    face_width = analyzer.get_distance(coords['jaw_left'], coords['jaw_right'])
                    eye_left_width = analyzer.get_distance(coords['eye_left_inner'], coords['eye_left_outer'])
                    eye_right_width = analyzer.get_distance(coords['eye_right_inner'], coords['eye_right_outer'])
                    avg_eye_width = (eye_left_width + eye_right_width) / 2
                    eye_size_ratio = avg_eye_width / face_width if face_width > 0 else 0

                    # 顔タイプ診断
                    face_type = analyzer.diagnose_face_type(adult_score, curve_score, straight_score, eye_size_ratio)
                    
                    # 結果表示
                    st.success(f"**診断結果: {face_type}**")
                    st.info(analyzer.get_face_type_description(face_type))
                    
                    if show_diagnostic_process:
                        st.subheader("🔍 診断プロセス")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**大人顔・子供顔分析**")
                            st.metric("大人/子供要素スコア", f"{adult_score}/5" if adult_score > child_score else f"{child_score}/5")
                            st.write("判定:", "大人顔" if adult_score >= 3 else "子供顔")
                            
                            if show_detailed_analysis:
                                for feature, result in adult_analysis.items():
                                    st.caption(f"• {result}")
                        
                        with col_b:
                            st.write("**曲線・直線要素分析**")
                            st.metric("曲線/直線要素スコア", f"{curve_score}/5 {straight_score}/5")
                            st.write("判定:", "曲線タイプ" if curve_score > 3 else ("直線タイプ" if straight_score > 3 else "中間タイプ"))

                            if show_detailed_analysis:
                                for feature, result in curve_analysis.items():
                                    st.caption(f"• {result}")
                        
                        st.write(f"**目の大きさ比率**: {eye_size_ratio:.3f} ({'大きめ' if eye_size_ratio > 0.26 else '小さめ'}) 基準値: 0.26")
                        
                        # 追加：詳細な数値データ表
                        if show_detailed_analysis:
                            st.subheader("📊 測定値詳細")
                            
                            # 基本寸法データ
                            face_height = analyzer.get_distance(coords['nose_root'], coords['chin'])
                            face_width = analyzer.get_distance(coords['jaw_left'], coords['jaw_right'])
                            
                            measurement_data = {
                                '項目': [
                                    '顔の高さ', '顔の幅', '顔の縦横比',
                                    '目の間隔', '目間隔比', '左目幅', '右目幅', '平均目幅', '目サイズ比',
                                    '鼻先から顎', 'あご比率', '鼻幅', '鼻/目比',
                                    '左目縦横比', '唇の厚み', '唇厚比'
                                ],
                                '測定値': [
                                    f"{face_height:.4f}", f"{face_width:.4f}", f"{face_width/face_height:.4f}",
                                    f"{analyzer.get_distance(coords['eye_left_inner'], coords['eye_right_inner']):.4f}",
                                    f"{analyzer.get_distance(coords['eye_left_inner'], coords['eye_right_inner'])/face_width:.4f}",
                                    f"{analyzer.get_distance(coords['eye_left_inner'], coords['eye_left_outer']):.4f}",
                                    f"{analyzer.get_distance(coords['eye_right_inner'], coords['eye_right_outer']):.4f}",
                                    f"{avg_eye_width:.4f}", f"{eye_size_ratio:.4f}",
                                    f"{analyzer.get_distance(coords['nose_tip'], coords['chin']):.4f}",
                                    f"{analyzer.get_distance(coords['nose_tip'], coords['chin'])/face_height:.4f}",
                                    f"{analyzer.get_distance(coords['nose_left'], coords['nose_right']):.4f}",
                                    f"{analyzer.get_distance(coords['nose_left'], coords['nose_right'])/avg_eye_width:.4f}",
                                    f"{analyzer.get_distance(coords['eye_left_top'], coords['eye_left_bottom'])/analyzer.get_distance(coords['eye_left_inner'], coords['eye_left_outer']):.4f}",
                                    f"{analyzer.get_distance(coords['upper_lip'], coords['lower_lip']):.4f}",
                                    f"{analyzer.get_distance(coords['upper_lip'], coords['lower_lip'])/face_height:.6f}"
                                ]
                            }
                            
                            df_measurements = pd.DataFrame(measurement_data)
                            st.dataframe(df_measurements, use_container_width=True)
                    
                    break
            else:
                st.warning("顔が検出されませんでした。顔がはっきり写った画像をアップロードしてください。")
    
    # リアルタイム診断
    if run:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("カメラを開けませんでした。カメラが接続されているか確認してください。")
            return
        
        # FPS制限用
        fps_limit = 5  # 診断精度を上げるため低めに設定
        prev_time = time.time()
        
        while st.session_state["run"] and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("フレームを読み取れませんでした。")
                break
            
            # FPS制限
            current_time = time.time()
            if current_time - prev_time < 1.0 / fps_limit:
                time.sleep(0.01)
                continue

            prev_time = current_time
            
            # フレームを左右反転（鏡像）
            frame = cv2.flip(frame, 1)
            
            # BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipeで顔検出・分析
            results = analyzer.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 顔の傾きチェック
                    landmarks = face_landmarks.landmark
                    eye_left = landmarks[33]
                    eye_right = landmarks[263]
                    dx = eye_right.x - eye_left.x
                    dy = eye_right.y - eye_left.y
                    angle = math.degrees(math.atan2(dy, dx))

                    if abs(angle) > 15:
                        with diagnosis_placeholder.container():
                            st.warning(f"顔が傾いています（角度 {angle:.1f}°）。正面を向いてください。")
                        continue
                    
                    # 顔の特徴点描画（オプション）
                    if show_landmarks:
                        mp_drawing.draw_landmarks(
                            rgb_frame,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_CONTOURS,
                            None,
                            mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                    
                    # 分析実行
                    adult_score, child_score, adult_analysis = analyzer.analyze_adult_child_features(face_landmarks.landmark)
                    curve_score, straight_score, curve_analysis = analyzer.analyze_curve_straight_features(face_landmarks.landmark)

                    # 目の大きさ比率を計算
                    coords = {k: (face_landmarks.landmark[v].x, face_landmarks.landmark[v].y) for k, v in analyzer.landmark_indices.items()}
                    face_width = analyzer.get_distance(coords['jaw_left'], coords['jaw_right'])
                    eye_left_width = analyzer.get_distance(coords['eye_left_inner'], coords['eye_left_outer'])
                    eye_right_width = analyzer.get_distance(coords['eye_right_inner'], coords['eye_right_outer'])
                    avg_eye_width = (eye_left_width + eye_right_width) / 2
                    eye_size_ratio = avg_eye_width / face_width if face_width > 0 else 0
                    
                    # 顔タイプ診断
                    face_type = analyzer.diagnose_face_type(adult_score, curve_score, eye_size_ratio)
                    
                    # 結果表示
                    with diagnosis_placeholder.container():
                        st.success(f"**診断結果: {face_type}**")
                        st.info(analyzer.get_face_type_description(face_type))
                        
                        # セッションに結果を保存
                        st.session_state["last_result"] = face_type
                    
                    # 詳細分析表示
                    if show_diagnostic_process:
                        with analysis_placeholder.container():
                            st.subheader("🔍 診断プロセス")
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.write("**大人顔・子供顔分析**")
                                st.metric("大人/子供要素スコア", f"{adult_score}/5" if adult_score > child_score else f"{child_score}/5")
                                st.write("判定:", "大人顔" if adult_score >= 3 else "子供顔")
                            
                                if show_detailed_analysis:
                                    for feature, result in adult_analysis.items():
                                        st.caption(f"• {result}")
                        
                            with col_b:
                                st.write("**曲線・直線要素分析**")
                                st.metric("曲線/直線要素スコア", f"{curve_score}/5 {straight_score}/5")
                                st.write("判定:", "曲線タイプ" if curve_score > 3 else ("直線タイプ" if straight_score > 3 else "中間タイプ"))
                                
                                if show_detailed_analysis:
                                    for feature, result in curve_analysis.items():
                                        st.caption(f"• {result}")
                            
                            st.write(f"**目の大きさ比率**: {eye_size_ratio:.3f} ({'大きめ' if eye_size_ratio > 0.26 else '小さめ'}) 基準値: 0.26")
                            
                            # リアルタイム用の簡易数値表示
                            if show_detailed_analysis:
                                st.write("**主要測定値**")
                                face_height = analyzer.get_distance(coords['nose_root'], coords['chin'])
                                face_width = analyzer.get_distance(coords['jaw_left'], coords['jaw_right'])
                                st.caption(f"• 顔縦横比: {face_width/face_height:.3f}")
                                st.caption(f"• 目間隔比: {analyzer.get_distance(coords['eye_left_inner'], coords['eye_right_inner'])/face_width:.3f}")
                                st.caption(f"• あご比: {analyzer.get_distance(coords['nose_tip'], coords['chin'])/face_height:.3f}")
                                st.caption(f"• 鼻/目比: {analyzer.get_distance(coords['nose_left'], coords['nose_right'])/avg_eye_width:.3f}")
                    
                    break
            else:
                with diagnosis_placeholder.container():
                    st.info("顔が検出されていません。カメラの前に顔を向けてください。")
                analysis_placeholder.empty()
            
            # フレーム表示
            FRAME_WINDOW.image(rgb_frame)
        
        cap.release()
        st.session_state["run"] = False

    else:
        st.info("「カメラを開始」をクリックして診断を開始してください。")

if __name__ == "__main__":
    main()