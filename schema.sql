PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS test_questions (
    question TEXT NOT NULL,
    question_uuid TEXT PRIMARY KEY,
    explanation TEXT
);

CREATE TABLE IF NOT EXISTS test_answers (
    question_uuid TEXT NOT NULL,
    option TEXT NOT NULL,
    true_or_false INTEGER NOT NULL,
    FOREIGN KEY(question_uuid) REFERENCES test_questions(question_uuid)
);
