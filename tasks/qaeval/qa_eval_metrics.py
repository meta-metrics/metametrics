import regex
import string

class QA_Metrics:
    # EVOUNA Default Metrics
    @staticmethod
    def normalize_answer(s):
        def remove_articles(text):
            return regex.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return QA_Metrics.normalize_answer(prediction) == QA_Metrics.normalize_answer(ground_truth)

    @staticmethod
    def ems(prediction, ground_truths):
        return max([QA_Metrics.exact_match_score(prediction, gt) for gt in ground_truths])

    @staticmethod
    def lexical_match_score(prediction, ground_truth):
        return QA_Metrics.normalize_answer(ground_truth) in QA_Metrics.normalize_answer(prediction)

    @staticmethod
    def lms(prediction, ground_truths):
        return max([QA_Metrics.lexical_match_score(prediction, gt) for gt in ground_truths])
