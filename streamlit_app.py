import os
import streamlit as st
from crewai import Crew, Agent, Task
from dotenv import load_dotenv
from crewai_tools import SeleniumScrapingTool, ScrapeWebsiteTool
from langchain_openai import ChatOpenAI
from search_tools import SearchTools
import sys
import time


load_dotenv()

# Define the correct username and password
CORRECT_USERNAME = os.getenv("YOUR_APP_USERNAME")
CORRECT_PASSWORD = os.getenv("YOUR_APP_PASSWORD")

def authenticate(username, password):
    return username == CORRECT_USERNAME and password == CORRECT_PASSWORD

class Crew1:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.7)
        self.scrape_tool = ScrapeWebsiteTool()
        self.selenium_tool = SeleniumScrapingTool()

    def run(self, search_terms, website_url, researcher_task_description, blog_writer_task_description):
        researcher_scraper_agent = Agent(
            role=" Expert Researcher and Summarizer",
            backstory=("""You are and expert in researching and summarizing news. Send summarized content to Writer Agent."""),
            goal=("""research and scrape content and provide summarized content to the writer agent."""),
            tools=[SearchTools.search_internet, self.selenium_tool, self.scrape_tool],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
            memory=True,
        )

        blog_writer_agent = Agent(
            role="Expert Blog Writer",
            backstory=("""You are a renowned content creator, known for your thoughtful and insightful articles. 
                                 You transform complicated content into easy to understand narratives. 
                                 You are tasked with writing blog content based on the given search term and text content sent to you by “Researcher Agent. 
                                 After blog is complete send to Social Media Influencer agent."""),
            goal= ("""Craft compelling blog content on the given search term and the content provided to you by the researcher agent."""),
            tools=[SearchTools.search_internet],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

        scrape_task = Task(
            agent=researcher_scraper_agent,
            description=researcher_task_description,
            expected_output=f"Full content extracted from the search results based on the provided search terms {search_terms}.",
            search_terms=search_terms,
            website_url=website_url
        )

        craft_task = Task(
            agent=blog_writer_agent,
            description=blog_writer_task_description,
            expected_output="Compelling blog content, should contain atleast 500 words.",
        )

        crew = Crew(
            agents=[researcher_scraper_agent, blog_writer_agent],
            tasks=[scrape_task, craft_task],
            verbose=True,
        )

        return crew.kickoff()

class Crew2:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.7)

    def run(self, blog_content):
        social_media_influencer_agent = Agent(
            role="Social Influencer, content summarizer and social media poster",
            backstory=("""You are an Expert Vetted social media influencer and manager. You are known for your sleuthing abilities and your ability to generate complicated content sent from Agent Writer into Social media posts in 280 characters or less."""),
            goal=("""summarize and Craft a compelling social media post from the blog content porovided by the writer agent"""),
            tools=[SearchTools.search_internet],
            allow_delegation=False,
            memory=True,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

        generate_social_media_posts_task = Task(
            description=("""    
                **Task**: Generate Social Media Posts
                **Description**: Generate quirky and engaging social media posts from the content provided by the Blog/Writer agent.
            """),
            expected_output="Social Media content of 280 characters based on blog content.",
            agent=social_media_influencer_agent,
        )

        crew = Crew(
            agents=[social_media_influencer_agent],
            tasks=[generate_social_media_posts_task],
            verbose=True,
        )

        return crew.kickoff()

def main():
    st.title("Turtles Crew-AI")

    # Check if the user is logged in
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        # Ask for username and password
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
            else:
                st.error("Incorrect username or password. Please try again.")
                return

    # If not logged in, return early to prevent displaying the app
    if not st.session_state.logged_in:
        return

    st.sidebar.title("Research and Blog Team")
    search_terms = st.sidebar.text_input("Search Terms")
    website_url = st.sidebar.text_input("Website URL (optional)")
    researcher_task_description = st.sidebar.text_area("Researcher Task Description")
    blog_writer_task_description = st.sidebar.text_area("Blog Writer Task Description")

    if st.sidebar.button("Run Crew 1"):
        crew1 = Crew1()
        crew1_result = crew1.run(search_terms, website_url, researcher_task_description, blog_writer_task_description)
        st.subheader("Crew 1 Result")
        st.write(crew1_result)

    st.sidebar.title("Social Media Team")
    blog_content = st.sidebar.text_area("Blog Content")

    if st.sidebar.button("Run Crew 2"):
        crew2 = Crew2()
        crew2_result = crew2.run(blog_content)
        st.subheader("Crew 2 Result")
        st.write(crew2_result)

if __name__ == "__main__":
    main()
