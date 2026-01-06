import streamlit as st
import json
import os
import base64
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


PROMPTS_FILE = "prompts.json"
RESULTS_FILE = "results.json"

AGENT_NAMES = [
    "Agent 1 - XPER Control",
    "Agent 2 - Moderation",
    "Agent 3 - Nudity Control",
    "Agent 4 - Typo Fixed",
    "Agent 5 - Labeling"
]

DEFAULT_PROMPTS = {
    "Agent 1 - XPER Control": "Bu agent XPER kontrolu yapar. Icerigi analiz ederek XPER standartlarina uygunlugunu kontrol eder.",
    "Agent 2 - Moderation": "Bu agent moderasyon kontrolu yapar. Icerigin topluluk kurallarina uygunlugunu denetler.",
    "Agent 3 - Nudity Control": "Bu agent nudity kontrolu yapar. Eğer görsel varsa görselde uygunsuz içerik olup olmadığı kontrol eder. ",
    "Agent 4 - Typo Fixed": "Bu agent yazim hatalarini duzeltir. Metindeki imla ve dilbilgisi hatalarini tespit edip duzeltir.",
    "Agent 5 - Labeling": "Bu agent etiketleme yapar. Icerigi uygun kategorilere ve etiketlere gore siniflandirir."
}

XPER_LEVEL_DESCRIPTIONS = {
    1: "",
    2: "",
    3: "",
    4: "",
    5: "",
    6: "",
    7: "",
    8: "",
    9: "",
    10: ""
}


def load_prompts():
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return DEFAULT_PROMPTS.copy()
    return DEFAULT_PROMPTS.copy()


def save_prompts(prompts):
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)


def load_results():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def save_results(results):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def encode_image_to_base64(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        return base64.b64encode(bytes_data).decode("utf-8")
    return None


def get_image_mime_type(uploaded_file):
    if uploaded_file is not None:
        return uploaded_file.type
    return "image/jpeg"


def call_openai_api(api_key, messages):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-5.1",
        "messages": messages,
        "max_completion_tokens": 4096,
        "temperature": 0.7
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"API Hatasi: {response.status_code} - {response.text}")

    result = response.json()
    return result["choices"][0]["message"]["content"]


def run_single_agent(api_key, agent_name, prompt, user_input, previous_output, xper_level, image_base64=None,
                     image_mime_type=None):
    is_nudity_agent = agent_name == "Agent 3 - Nudity Control"

    if is_nudity_agent:
        if image_base64:
            user_content = "Bu gorevde SADECE yuklenen gorseli analiz et."
        else:
            user_content = "Bu istekte gorsel yok. Nudity kontrolu yapilamadi."
    else:
        if previous_output:
            user_content = f"Onceki agent ciktisi:\n{previous_output}\n\nKullanici girdisi:\n{user_input}"
        else:
            user_content = f"Kullanici girdisi:\n{user_input}"

    xper_context = f"""
XPER Seviyesi: {xper_level}/10
XPER Seviye Aciklamasi: {XPER_LEVEL_DESCRIPTIONS[xper_level]}


Bu XPER seviyesine ({xper_level}) uygun sekilde analiz yap ve karar ver.
"""

    system_message = f"{prompt}\n\n{xper_context}"

    if previous_output:
        user_content = f"Onceki agent ciktisi:\n{previous_output}\n\nKullanici girdisi:\n{user_input}"
    else:
        user_content = f"Kullanici girdisi:\n{user_input}"

    messages = [
        {"role": "system", "content": system_message}
    ]

    if image_base64 and image_mime_type:
        user_message_content = [
            {"type": "text", "text": user_content},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime_type};base64,{image_base64}"
                }
            }
        ]
        messages.append({"role": "user", "content": user_message_content})
    else:
        messages.append({"role": "user", "content": user_content})

    try:
        result = call_openai_api(api_key, messages)
        return result, None
    except Exception as e:
        return None, str(e)


def init_session_state():
    if "prompts" not in st.session_state:
        st.session_state.prompts = load_prompts()

    if "current_step" not in st.session_state:
        st.session_state.current_step = -1

    if "agent_outputs" not in st.session_state:
        st.session_state.agent_outputs = {}

    if "pipeline_status" not in st.session_state:
        st.session_state.pipeline_status = "idle"

    if "user_input_saved" not in st.session_state:
        st.session_state.user_input_saved = ""

    if "image_data" not in st.session_state:
        st.session_state.image_data = None

    if "image_mime" not in st.session_state:
        st.session_state.image_mime = None

    if "api_key_saved" not in st.session_state:
        st.session_state.api_key_saved = ""

    if "error_message" not in st.session_state:
        st.session_state.error_message = None

    if "xper_level" not in st.session_state:
        st.session_state.xper_level = 5

    if "rejected_by_agent" not in st.session_state:
        st.session_state.rejected_by_agent = None


def reset_pipeline():
    st.session_state.current_step = -1
    st.session_state.agent_outputs = {}
    st.session_state.pipeline_status = "idle"
    st.session_state.error_message = None
    st.session_state.rejected_by_agent = None


def start_pipeline(api_key, user_input, image_base64, image_mime, xper_level):
    st.session_state.api_key_saved = OPENAI_API_KEY

    st.session_state.user_input_saved = user_input
    st.session_state.image_data = image_base64
    st.session_state.image_mime = image_mime
    st.session_state.xper_level = xper_level
    st.session_state.current_step = 0
    st.session_state.agent_outputs = {}
    st.session_state.pipeline_status = "running"
    st.session_state.error_message = None


def run_next_agent():
    step = st.session_state.current_step

    if step >= len(AGENT_NAMES):
        st.session_state.pipeline_status = "completed"
        save_current_result()
        return

    agent_name = AGENT_NAMES[step]
    prompt = st.session_state.prompts.get(agent_name, DEFAULT_PROMPTS[agent_name])

    if step > 0:
        prev_agent = AGENT_NAMES[step - 1]
        previous_output = st.session_state.agent_outputs.get(prev_agent, "")
    else:
        previous_output = None

    output, error = run_single_agent(
        api_key=st.session_state.api_key_saved,
        agent_name=agent_name,
        prompt=prompt,
        user_input=st.session_state.user_input_saved,
        previous_output=previous_output,
        xper_level=st.session_state.xper_level,
        image_base64=st.session_state.image_data,
        image_mime_type=st.session_state.image_mime
    )

    if error:
        st.session_state.error_message = f"{agent_name} hatasi: {error}"
        st.session_state.pipeline_status = "error"
        return

    st.session_state.agent_outputs[agent_name] = output
    if st.session_state.rejected_by_agent is None:
        if is_negative_decision(output):
            st.session_state.rejected_by_agent = agent_name

    st.session_state.current_step = step + 1

    if st.session_state.current_step >= len(AGENT_NAMES):
        st.session_state.pipeline_status = "completed"
        save_current_result()

def is_negative_decision(output_text: str) -> bool:
    if not output_text:
        return False

    lowered = output_text.lower()

    negative_signals = [
        "reddedildi",
        "kaldırıldı",
        "silindi",
        "yayınlanamaz",
        "yayına alınamaz",
        "uygun değildir",
        "uygun değil",
        "yasaya aykırı",
        "ahlaka aykırı",

        "kendine zarar",
        "intihar"
        "cinsellik",

        "seviye gerektirmektedir",
        "profil seviyenizi",
        "erişim yetkiniz yok",
        "yetkiniz bulunmamaktadır",
        "daha fazla bilgi için",
        "erişim kısıtı",
        "sadece belirli seviyeler"
    ]

    return any(signal in lowered for signal in negative_signals)


def save_current_result():
    result_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": st.session_state.user_input_saved,
        "xper_level": st.session_state.xper_level,
        "had_image": st.session_state.image_data is not None,
        "outputs": st.session_state.agent_outputs.copy()
    }
    results = load_results()
    results.insert(0, result_entry)
    results = results[:50]
    save_results(results)


def render_agent_status():
    st.subheader("Pipeline Durumu")

    current = st.session_state.current_step
    status = st.session_state.pipeline_status

    for i, agent_name in enumerate(AGENT_NAMES):
        if status == "idle":
            icon = "( )"
            state = "Bekliyor"
            color = "gray"
        elif i < current:
            icon = "(+)"
            state = "Tamamlandi"
            color = "green"
        elif i == current and status == "running":
            icon = "(>)"
            state = "Calisiyor..."
            color = "orange"
        elif i == current and status == "error":
            icon = "(X)"
            state = "Hata!"
            color = "red"
        else:
            icon = "( )"
            state = "Bekliyor"
            color = "gray"

        st.markdown(f":{color}[**{icon} {agent_name}**: {state}]")

    if status == "completed":
        if st.session_state.rejected_by_agent:
            st.error(
                f"İçerik reddedildi. İlk red aldığı aşama: {st.session_state.rejected_by_agent}"
            )
        else:
            st.success("İçerik yayınlanabilir")

    elif status == "error" and st.session_state.error_message:
        st.error(st.session_state.error_message)


def main():
    st.set_page_config(
        page_title="Agent Pipeline Test",
        layout="wide"
    )

    init_session_state()

    st.title("Agent Pipeline Test Sistemi")
    st.markdown("5 agentli sirayla calisan pipeline test araci")

    tab1, tab2, tab3 = st.tabs(["Pipeline Calistir", "Prompt Ayarlari", "Gecmis Sonuclar"])

    with tab1:
        st.header("Pipeline Calistirma")

        col1, col2 = st.columns([1, 1])

        with col1:


            user_input = st.text_area(
                "Kullanici Girdisi",
                height=150,
                placeholder="Analiz edilecek metni buraya girin...",
                disabled=st.session_state.pipeline_status == "running"
            )

            st.subheader("XPER Seviyesi")
            xper_level = st.slider(
                "Kontrol hassasiyetini secin",
                min_value=1,
                max_value=10,
                value=st.session_state.xper_level,
                disabled=st.session_state.pipeline_status == "running",
                help="1: En gevsek kontrol, 10: En siki kontrol"
            )

            st.info(f"Secilen seviye: {xper_level} - {XPER_LEVEL_DESCRIPTIONS[xper_level]}")

            uploaded_image = st.file_uploader(
                "Gorsel Ekle (Opsiyonel)",
                type=["png", "jpg", "jpeg", "gif", "webp"],
                disabled=st.session_state.pipeline_status == "running"
            )

            if uploaded_image:
                st.image(uploaded_image, caption="Yuklenen Gorsel", use_container_width=True)

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                start_btn = st.button(
                    "Pipeline Baslat",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.pipeline_status == "running"
                )

            with col_btn2:
                reset_btn = st.button(
                    "Sifirla",
                    type="secondary",
                    use_container_width=True
                )

        with col2:
            render_agent_status()

            if st.session_state.pipeline_status != "idle":
                st.markdown(f"**Aktif XPER Seviyesi:** {st.session_state.xper_level}/10")

            total = len(AGENT_NAMES)
            completed = st.session_state.current_step if st.session_state.current_step >= 0 else 0
            if st.session_state.pipeline_status == "completed":
                completed = total

            progress_value = completed / total if total > 0 else 0
            st.progress(progress_value, text=f"Ilerleme: {completed}/{total}")

        if start_btn:
            if not user_input:
                st.error("Lutfen kullanici girdisi girin!")
            else:
                image_base64 = None
                image_mime = None
                if uploaded_image:
                    image_base64 = encode_image_to_base64(uploaded_image)
                    image_mime = get_image_mime_type(uploaded_image)

                start_pipeline(OPENAI_API_KEY, user_input, image_base64, image_mime, xper_level)


                st.rerun()

        if reset_btn:
            reset_pipeline()
            st.rerun()

        if st.session_state.pipeline_status == "running":
            run_next_agent()
            st.rerun()

        if st.session_state.agent_outputs:
            st.divider()
            st.subheader("Agent Ciktilari")

            for agent_name in AGENT_NAMES:
                if agent_name in st.session_state.agent_outputs:
                    with st.expander(f"{agent_name}", expanded=True):
                        st.markdown(st.session_state.agent_outputs[agent_name])

    with tab2:
        st.header("Prompt Ayarlari")
        st.markdown("Her agent icin promptlari duzenleyebilirsiniz. Degisiklikler otomatik olarak kaydedilir.")

        st.warning(
            "Not: XPER seviyesi bilgisi otomatik olarak her agentin promptuna eklenir. Promptlarda XPER seviyesini manuel olarak belirtmenize gerek yok.")

        prompts_changed = False

        for agent_name in AGENT_NAMES:
            st.subheader(agent_name)

            current_prompt = st.session_state.prompts.get(agent_name, DEFAULT_PROMPTS[agent_name])

            new_prompt = st.text_area(
                f"{agent_name} Promptu",
                value=current_prompt,
                height=150,
                key=f"prompt_{agent_name}",
                label_visibility="collapsed"
            )

            if new_prompt != current_prompt:
                st.session_state.prompts[agent_name] = new_prompt
                prompts_changed = True

            st.divider()

        if prompts_changed:
            save_prompts(st.session_state.prompts)
            st.success("Promptlar kaydedildi!")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Tum Promptlari Sifirla", type="secondary"):
                st.session_state.prompts = DEFAULT_PROMPTS.copy()
                save_prompts(st.session_state.prompts)
                st.success("Promptlar varsayilan degerlere sifirlandi!")
                st.rerun()

        with col2:
            if st.button("Promptlari Dosyadan Yukle", type="secondary"):
                st.session_state.prompts = load_prompts()
                st.success("Promptlar dosyadan yuklendi!")
                st.rerun()

    with tab3:
        st.header("Gecmis Sonuclar")

        results = load_results()

        if not results:
            st.info("Henuz kayitli sonuc bulunmuyor.")
        else:
            if st.button("Tum Gecmisi Temizle", type="secondary"):
                save_results([])
                st.success("Gecmis temizlendi!")
                st.rerun()

            for i, result in enumerate(results):
                timestamp = result.get("timestamp", "Bilinmiyor")
                try:
                    dt = datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime("%d/%m/%Y %H:%M:%S")
                except:
                    formatted_time = timestamp

                had_image = result.get("had_image", False)
                image_indicator = " [Gorsel Var]" if had_image else ""
                xper_lvl = result.get("xper_level", "?")

                with st.expander(f"Sonuc {i + 1} - {formatted_time} - XPER: {xper_lvl}/10{image_indicator}"):
                    st.markdown(f"**XPER Seviyesi:** {xper_lvl}/10")
                    st.markdown("**Kullanici Girdisi:**")
                    st.text(result.get("user_input", "")[:500])

                    st.markdown("**Agent Ciktilari:**")
                    outputs = result.get("outputs", {})
                    for agent_name in AGENT_NAMES:
                        if agent_name in outputs:
                            st.markdown(f"**{agent_name}:**")
                            st.markdown(outputs[agent_name])
                            st.divider()


if __name__ == "__main__":
    main()




