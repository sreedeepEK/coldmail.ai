import chromadb
import uuid
from PyPDF2 import PdfReader


class Portfolio:
    def __init__(self, file_path="data/SreedeepEK_Resume.pdf"):
        self.file_path = file_path
        self.data = self.extract_pdf_text()  # Read and extract PDF content
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def extract_pdf_text(self):
        """Extracts text from the PDF file."""
        reader = PdfReader(self.file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()  # Clean up any extra whitespace

    def parse_pdf_text(self):
        """Parses raw PDF text into a structured format."""
        sections = self.data.split("\n\n")  # Split text into sections
        parsed_data = []

        for section in sections:
            if "Techstack:" in section and "Links:" in section:  # Assuming structured sections
                try:
                    techstack = section.split("Techstack:")[1].split("Links:")[0].strip()
                    links = section.split("Links:")[1].strip()
                    parsed_data.append({"techstack": techstack, "links": links})
                except IndexError:
                    continue  # Skip malformed sections
        return parsed_data

    def load_portfolio(self):
        """Loads portfolio into ChromaDB."""
        parsed_data = self.parse_pdf_text()
        if not self.collection.count():
            for entry in parsed_data:
                self.collection.add(
                    documents=[entry["techstack"]],  # Ensure techstack is a single string
                    metadatas={"links": entry["links"]},
                    ids=[str(uuid.uuid4())]
                )

    def query_links(self, skills):
        """Queries portfolio for relevant links based on skills."""
        if isinstance(skills, list):
            # Join the skills into a single string
            skills = " ".join(skills)

        # Query ChromaDB with a single query string
        return self.collection.query(query_texts=[skills], n_results=2).get('metadatas', [])
