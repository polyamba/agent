from typing import TypedDict, Annotated, List, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
import re
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0,
)

class InterviewLog:
    """Class to manage interview session logging"""
    
    def __init__(self, participant_name: str):
        self.data = {
            "participant_name": participant_name,
            "session_start": datetime.now().isoformat(),
            "turns": [],
            "final_feedback": ""
        }
        self.current_turn_id = 0
    
    def add_turn(
        self, 
        agent_visible_message: str, 
        user_message: str, 
        internal_thoughts: str
    ):
        """Add a new turn to the interview log"""
        self.current_turn_id += 1
        self.data["turns"].append({
            "turn_id": self.current_turn_id,
            "agent_visible_message": agent_visible_message,
            "user_message": user_message,
            "internal_thoughts": internal_thoughts
        })
    
    def set_final_feedback(self, feedback: str):
        """Set the final feedback summary"""
        self.data["final_feedback"] = feedback
        self.data["session_end"] = datetime.now().isoformat()
    
    def save(self, filename: str = "interview_log.json"):
        """Save the log to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logger.info(f"Interview log saved to {filename}")
        return filename

class InterviewState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    company: str
    position: str
    current_difficulty: Literal["easy", "medium", "hard"]
    score_history: List[float]
    total_questions: int
    max_questions: int
    company_research: str
    current_question: str
    candidate_answer: str
    feedback: str
    should_continue: bool
    
    reflection_log: List[str]
    agent_instructions: str
    
    conversation_memory: List[dict]
    covered_topics: List[str]
    memory_window: int
    
    participant_name: str
    interview_log: Optional[InterviewLog]
    current_internal_thoughts: str  

def adjust_difficulty(current: str, score: float) -> str:
    """Adaptive difficulty adjustment"""
    levels = {"easy": 0, "medium": 1, "hard": 2}
    reverse_levels = {0: "easy", 1: "medium", 2: "hard"}
    
    if score >= 0.9:
        if current == "hard":
            return "hard"
        return reverse_levels[levels[current] + 1]
    elif score >= 0.7:
        return current
    else:
        if current == "easy":
            return "easy"
        return reverse_levels[levels[current] - 1]



def greeting_node(state: InterviewState) -> dict:
    """Greet the participant and initialize logging"""
    name = state.get('participant_name', 'Candidate')
    
    interview_log = InterviewLog(name)
    
    greeting = f"Привет, {name}! Добро пожаловать на собеседование."
    greeting += f" Мы будем обсуждать позицию {state['position']} в компании {state['company']}."
    
    print(f"\n{greeting}")
    
    internal = f"[Observer]: Начало сессии. [Interviewer]: Подготовка к собеседованию для {name}."
    
    return {
        "interview_log": interview_log,
        "current_internal_thoughts": internal,
        "messages": [AIMessage(content=greeting)]
    }


def research_node(state: InterviewState) -> dict:
    """Research company (once at start)"""
    prompt = f"""
    TASK
    1) Проведи анализ компании {state['company']} для позиции {state['position']}. 
    2) Найди требуемый стек технологий для конкретной позиции {state['position']}.
    3) Запиши популярные темы вопросов для собеседования. 
    """
    result = llm.invoke(prompt)
    
    logger.info(f"[RESEARCH] Company: {state['company']}, Position: {state['position']}")
    
    return {
        "company_research": result.content,
        "messages": [AIMessage(content=f"Research: {result.content}")]
    }


def reflection_node(state: InterviewState) -> dict:
    """
    Internal dialogue before generating response.
    Agent 1 (Observer) evaluates user input.
    Agent 2 (Interviewer) receives instructions.
    """
    candidate_answer = state.get('candidate_answer', '')
    current_question = state.get('current_question', '')
    score_history = state.get('score_history', [])
    conversation_memory = state.get('conversation_memory', [])
    
    memory_context = ""
    if conversation_memory:
        memory_context = "\n".join([
            f"Q: {m['question'][:50]}... A: {m['answer'][:50]}..."
            for m in conversation_memory[-3:]
        ])
    
    observer_prompt = f"""
    Ты — агент-критик. Твоя задача — проанализировать ответ кандидата.
    
    Текущий вопрос: {current_question}
    Ответ кандидата: {candidate_answer}
    История оценок: {score_history}
    Предыдущий контекст: {memory_context}
    
    Предоставь анализ строго в следующем формате:
    OBSERVATION: [твои выводы о качестве ответа кандидата]
    KNOWLEDGE_GAPS: [какие пробелы, ошибки или неточности были замечены]
    INSTRUCTION: [на чем конкретно должен сфокусироваться следующий вопрос. Не повторяй пройденные темы]
    REASONING: [логическое обоснование: нужно ли углубиться в тему из-за неуверенности кандидата или перейти к новой]
    """
    
    observer_result = llm.invoke(observer_prompt)
    observer_output = observer_result.content
    
    interviewer_prompt = f"""
    Ты — интервьюер. Твоя задача — определить следующий вопрос, основываясь на анализе от агента-критика.
    
    Агент-критик:
    {observer_output}

    Сформулируй вопрос для кандидата.
    """
    
    interviewer_result = llm.invoke(interviewer_prompt)
    interviewer_output = interviewer_result.content
    
    internal_thoughts = f"[Observer]: {observer_output[:200]}... [Interviewer]: {interviewer_output[:200]}..."
    
    logger.info("=" * 60)
    logger.info("[HIDDEN REFLECTION - INTERNAL DIALOGUE]")
    logger.info("=" * 60)
    logger.info(f"Candidate answer: {candidate_answer[:100]}...")
    logger.info(f"OBSERVER:\n{observer_output}")
    logger.info(f"INTERVIEWER:\n{interviewer_output}")
    logger.info("=" * 60)
    
    instruction_match = re.search(r"INSTRUCTION:\s*(.+?)(?:\n|REASONING:|$)", interviewer_output, re.DOTALL)
    instruction = instruction_match.group(1).strip() if instruction_match else "Двинемся к следующей теме."
    
    reflection_entry = f"""
    [REFLECTION #{state['total_questions']}]
    Observer: {observer_output[:150]}...
    Interviewer: {instruction}
    """
    
    new_reflection_log = state.get('reflection_log', []) + [reflection_entry]
    
    return {
        "reflection_log": new_reflection_log,
        "agent_instructions": instruction,
        "current_internal_thoughts": internal_thoughts,
        "messages": []
    }


def create_question_node(state: InterviewState) -> dict:
    """Create question with context awareness - avoids repeating topics"""
    
    covered_topics = state.get('covered_topics', [])
    agent_instructions = state.get('agent_instructions', '')
    conversation_memory = state.get('conversation_memory', [])
    interview_log = state.get('interview_log')
    
    memory_context = ""
    if conversation_memory:
        last_n = conversation_memory[-(state.get('memory_window', 3)):]
        memory_context = "\nПРЕДЫДУЩЕЕ ВЗАИМОДЕЙСТВИЕ С КАНДИДАТОМ (НЕ ПОВТОРЯТЬ ВОПРОСЫ):\n"
        for i, mem in enumerate(last_n):
            memory_context += f"{i+1}. Topic: {mem.get('topic', 'unknown')} - Q: {mem['question'][:80]}...\n"
    
    covered_str = ", ".join(covered_topics) if covered_topics else "none yet"
    
    prompt = f"""
    Текущая сложность: {state['current_difficulty']}
    
    {memory_context}
    
    Пройденные темы (ЗАПРЕЩЕНО задвать по ним вопросы): {covered_str}
    
    Инструкция от интервьюера: {agent_instructions}
    
    Сгенерируй НОВЫЙ вопрос для кандидата на позицию {state['position']} следующего уровня сложности: {state['current_difficulty']}.
    
    ВАЖНЫЕ ПРАВИЛА:
    1. Следуй инструкции интервьюера, если она предоставлена.
    2. Если интервьюер решил, что можно остаться на той же теме, задай углубленный вопрос по текущей теме.
    3. В противном случае выбери новую тему для вопроса.
    
    Формат вывода (строго соблюдай структуру):
    TOPIC: [основная тема вопроса]
    Question: [текст вопроса]
    """
    
    result = llm.invoke(prompt)
    text = result.content
    
    topic_match = re.search(r"TOPIC:\s*(.+?)(?:\n|Question:)", text, re.DOTALL)
    topic = topic_match.group(1).strip().lower() if topic_match else "python general"
    
    q_match = re.search(r"Question:\s*(.+?)", text, re.DOTALL)
    
    question = q_match.group(1).strip() 
    
    if topic in covered_topics:
        logger.warning(f"[CONTEXT] Тема '{topic}' уже обсуждалась! Добавляю пометку...")
        topic = f"{topic}_advanced"
    
    new_covered = covered_topics + [topic]
    
    visible_message = f"Вопрос {state['total_questions'] + 1} ({state['current_difficulty'].upper()}): {question}"
    
    logger.info(f"[QUESTION GEN] Topic: {topic}, Difficulty: {state['current_difficulty']}")
    logger.info(f"[CONTEXT] Covered topics: {new_covered}")
    
    print(f"\nВопрос {state['total_questions'] + 1} ({state['current_difficulty'].upper()}):")
    print(f"Тема: {topic}")
    print(question)
    
    return {
        "current_question": question,
        "covered_topics": new_covered,
        "messages": [AIMessage(content=visible_message)]
    }


def get_answer_node(state: InterviewState) -> dict:
    """Get candidate's answer"""
    answer = input("\nВаш ответ: ")
    return {"candidate_answer": answer}


def evaluate_answer_node(state: InterviewState) -> dict:
    """Evaluate answer and log the turn"""
    
    interview_log = state.get('interview_log')
    internal_thoughts = state.get('current_internal_thoughts', '')
    
    prompt = f"""
    Вопрос: {state['current_question']}
    Ответ кандидата: {state['candidate_answer']}
    
    Оцени ответ от 0.0 до 1.0 (float). 1.0=абсолютно верно, 0.0=абсолютно неверно.
    
    ОБЯЗАТЕЛЬНО следуй формату:
    SCORE: 0.85
    FEEDBACK: [1-2 предожения ОБЯЗАТЕЛЬНО на РУССКОМ]
    """
    
    result = llm.invoke(prompt)
    text = result.content
    
    score_match = re.search(r"SCORE:\s*([\d.]+)", text, re.IGNORECASE)
    score = float(score_match.group(1)) if score_match else 0.5
    score = min(1.0, max(0.0, score))
    
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", text, re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else "Нет обратной связи"
    
    print(f"\nОценка: {score:.2f}")
    print(f"Обратная связь: {feedback}")
    
    if interview_log:
        visible_message = f"Вопрос {state['total_questions'] + 1}: {state['current_question']}"
        
        if state['total_questions'] > 0 and internal_thoughts:
            turn_thoughts = internal_thoughts
        else:
            turn_thoughts = f"[Observer]: Первый вопрос. [Interviewer]: Начинаем с базового уровня {state['current_difficulty']}."
        
        interview_log.add_turn(
            agent_visible_message=visible_message,
            user_message=state['candidate_answer'],
            internal_thoughts=turn_thoughts
        )
    
    new_memory_entry = {
        "question": state['current_question'],
        "answer": state['candidate_answer'],
        "topic": state.get('covered_topics', ['unknown'])[-1],
        "score": score
    }
    
    conversation_memory = state.get('conversation_memory', [])
    new_memory = conversation_memory + [new_memory_entry]
    
    memory_window = state.get('memory_window', 3)
    if len(new_memory) > memory_window:
        new_memory = new_memory[-memory_window:]
    
    new_history = state['score_history'] + [score]
    new_difficulty = adjust_difficulty(state['current_difficulty'], score)
    total_q = state['total_questions'] + 1
    should_continue = total_q < state['max_questions']
    
    logger.info(f"[EVALUATE] Score: {score:.2f}, New difficulty: {new_difficulty}")
    logger.info(f"[MEMORY] Stored {len(new_memory)} interactions in memory")
    
    return {
        "feedback": feedback,
        "score_history": new_history,
        "current_difficulty": new_difficulty,
        "total_questions": total_q,
        "should_continue": should_continue,
        "conversation_memory": new_memory,
        "interview_log": interview_log
    }


def finalize_node(state: InterviewState) -> dict:
    """Generate final feedback and save the interview log"""
    
    interview_log = state.get('interview_log')
    score_history = state.get('score_history', [])
    covered_topics = state.get('covered_topics', [])
    
    avg_score = sum(score_history) / len(score_history) if score_history else 0
    
    final_prompt = f"""
    Составь обще мнение о кандидате и предоставь ему свою обратную связь.
    
    Кандидат: {state.get('participant_name', 'Кандидат')}
    Позиция: {state['position']}
    Компания: {state['company']}
    Средняя оценка: {avg_score:.2f}
    Кол-во заданных вопросов: {state['total_questions']}
    Темы, покрытые в рамках вопросов кандидату: {', '.join(covered_topics)}
    Динамика оценки: {score_history}
    
    Формат обратной связи:
    1. Общее впечатление (1-2 предложения)
    2. Сильные стороны (2-3 пункта)
    3. Что нужно/можно подтянуть (2-3 пункта)
    4. Рекомендация (найм/на рассмотрении/отказ с обязательным предоставлением объснением причины)
    
    Формулируй свои мысли, используя профессиональную этику и лексику.
    """
    
    result = llm.invoke(final_prompt)
    final_feedback = result.content
    
    print("\n" + "=" * 60)
    print("СОБЕСЕДОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Итоговая оценка: {avg_score:.2f}")
    print(f"Вопросов отвечено: {state['total_questions']}")
    print(f"Пройденные темы: {', '.join(covered_topics)}")
    print("\nИТОГОВЫЙ ОТЗЫВ:")
    print(final_feedback)
    
    if interview_log:
        interview_log.set_final_feedback(final_feedback)
        
        json_file = interview_log.save("interview_log.json")
        
        print(f"\nЛоги сохранены: {json_file}")
    
    return {
        "feedback": final_feedback,
        "interview_log": interview_log
    }


def route_decision(state: InterviewState) -> str:
    """Router: continue or end"""
    if state['should_continue']:
        return "reflection"
    return "finalize"


graph = StateGraph(InterviewState)

graph.add_node("greeting", greeting_node)
graph.add_node("research", research_node)
graph.add_node("reflection", reflection_node)
graph.add_node("create_question", create_question_node)
graph.add_node("get_answer", get_answer_node)
graph.add_node("evaluate", evaluate_answer_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("greeting")
graph.add_edge("greeting", "research")
graph.add_edge("research", "create_question")
graph.add_edge("create_question", "get_answer")
graph.add_edge("get_answer", "evaluate")

graph.add_conditional_edges(
    "evaluate",
    route_decision,
    {
        "reflection": "reflection",
        "finalize": "finalize"
    }
)

graph.add_edge("reflection", "create_question")

graph.add_edge("finalize", END)

adaptive_interview = graph.compile()


if __name__ == "__main__":
    print("Starting Adaptive Interview Agent")
    print("=" * 60)
    
    # Get participant name
    participant_name = input("Введите ваше ФИО: ")
    if not participant_name.strip():
        participant_name = "Анатолий Анатольевич Анатольев"
    
    result = adaptive_interview.invoke({
        "participant_name": participant_name,
        "company": "Yandex",
        "position": "Python Backend Developer",
        "current_difficulty": "easy",
        "score_history": [],
        "total_questions": 0,
        "max_questions": 5,
        "messages": [],
        "should_continue": True,
        "company_research": "",
        "current_question": "",
        "candidate_answer": "",
        "feedback": "",
        "reflection_log": [],
        "agent_instructions": "",
        "conversation_memory": [],
        "covered_topics": [],
        "memory_window": 3,
        "interview_log": None,
        "current_internal_thoughts": ""
    })
    
    print("Интервью завершено!")
