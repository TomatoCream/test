from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional, List, Any
from datetime import date, datetime
import argparse
import json

# Pydantic Model Definitions

class Link(BaseModel):
    """Represents a hyperlink, typically for navigation."""
    href: HttpUrl = Field(description="URL for the link.")

class Links(BaseModel):
    """Container for navigation links, commonly used in paginated API responses."""
    next: Optional[Link] = Field(default=None, description="Link to the next page of results.")
    self: Link = Field(description="Link to the current page of results.")
    first: Optional[Link] = Field(default=None, description="Link to the first page of results.")
    last: Optional[Link] = Field(default=None, description="Link to the last page of results.")

class Districts(BaseModel):
    """Detailed district information for an address."""
    sectors: str = Field(description="Sector identifier for the district.")
    location: str = Field(description="Descriptive name of the location or broader area.")
    region: str = Field(description="Geographical region (e.g., 'Islandwide').")
    id: int = Field(description="Unique identifier for the district.")
    regionId: str = Field(description="Identifier for the region, often mirroring the region name.")

class Address(BaseModel):
    """Represents a physical address."""
    floor: Optional[str] = Field(default=None, description="Floor number of the address, if applicable.")
    postalCode: str = Field(description="Postal code of the address.")
    districts: Districts = Field(description="Detailed district information.")
    isOverseas: bool = Field(description="Flag indicating if the address is located overseas.")
    foreignAddress2: Optional[str] = Field(default=None, description="Second line of foreign address, if applicable.")
    street: str = Field(description="Street name and number.")
    overseasCountry: Optional[str] = Field(default=None, description="Country name, if the address is overseas.")
    block: Optional[str] = Field(default=None, description="Block number of the address. E.g., '391B'") # Sample had "391B", making it Optional for wider applicability
    lat: float = Field(description="Latitude coordinate of the address.")
    lng: float = Field(description="Longitude coordinate of the address.")
    unit: Optional[str] = Field(default=None, description="Unit number of the address, if applicable.")
    building: Optional[str] = Field(default=None, description="Name of the building. E.g., 'MARINA BAY FINANCIAL CENTRE'") # Sample had a value, making it Optional for wider applicability
    foreignAddress1: Optional[str] = Field(default=None, description="First line of foreign address, if applicable.")

class Skill(BaseModel):
    """Represents a skill required or associated with a job."""
    uuid: str = Field(description="Unique identifier for the skill.")
    skill: str = Field(description="Name of the skill (e.g., 'Oral & Written Communication Skills').")

class Status(BaseModel):
    """Represents the current status of a job posting."""
    id: str = Field(description="Identifier for the job status (e.g., '102').")
    jobStatus: str = Field(description="Descriptive status of the job posting (e.g., 'Re-open').")

class PositionLevel(BaseModel):
    """Represents the seniority or level of a job position."""
    position: str = Field(description="Description of the position level (e.g., 'Senior Management').")
    id: int = Field(description="Identifier for the position level.")

class SalaryType(BaseModel):
    """Specifies the type or frequency of salary payment."""
    salaryType: str = Field(description="Type of salary payment (e.g., 'Monthly').")

class Salary(BaseModel):
    """Represents salary information for a job."""
    maximum: int = Field(description="Maximum salary amount for the position.")
    minimum: int = Field(description="Minimum salary amount for the position.")
    type: SalaryType = Field(description="Details about the salary payment type.")

class Metadata(BaseModel):
    """Contains metadata related to a job posting."""
    jobDetailsUrl: HttpUrl = Field(description="URL to the full job details page on the job portal.")
    isPostedOnBehalf: bool = Field(description="Flag indicating if the job is posted on behalf of another company.")
    isHideHiringEmployerName: bool = Field(description="Flag indicating if the hiring employer's name is hidden from applicants.")
    jobPostId: str = Field(description="Unique identifier for the job post (e.g., 'MCF-2025-0714896').")
    totalNumberJobApplication: int = Field(description="Total number of applications received for this job posting.")
    isHideSalary: bool = Field(description="Flag indicating if the salary information is hidden from applicants.")
    newPostingDate: date = Field(description="Date when the job was newly posted or most recently reposted.")
    updatedAt: datetime = Field(description="Timestamp of the last update to the job posting information.")

class ResponsiveEmployer(BaseModel):
    """Indicates whether an employer is considered responsive."""
    isResponsive: bool = Field(description="Flag indicating if the employer is responsive to applications.")

class PostedCompany(BaseModel):
    """Details of the company that posted the job."""
    uen: str = Field(description="Unique Entity Number (UEN) of the company.")
    logoUploadPath: Optional[HttpUrl] = Field(default=None, description="Path or URL to the company's logo image.")
    responsiveEmployer: Optional[ResponsiveEmployer] = Field(default=None, description="Information about the employer's responsiveness.")
    logoFileName: Optional[str] = Field(default=None, description="Filename of the company's logo.")
    name: str = Field(description="Name of the company that posted the job.")

class EmploymentType(BaseModel):
    """Describes the type of employment."""
    id: int = Field(description="Identifier for the employment type.")
    employmentType: str = Field(description="Nature of the employment (e.g., 'Internship/Attachment', 'Full-time').")

class Category(BaseModel):
    """Represents the category of a job."""
    id: int = Field(description="Identifier for the job category.")
    category: str = Field(description="Name of the job category (e.g., 'Real Estate / Property Management').")

class JobResult(BaseModel):
    """Represents a single job listing details from the search results."""
    uuid: str = Field(description="Unique identifier for this job result entry.")
    address: Address = Field(description="Physical location details of the job.")
    schemes: List[Any] = Field(default_factory=list, description="List of applicable government schemes or programs. Structure of items is undefined from sample.")
    skills: Skill = Field(description="Primary skill associated with the job. Based on the sample, this is a single skill object.")
    status: Status = Field(description="Current status of the job posting.")
    title: str = Field(description="Title of the job position.")
    positionLevels: PositionLevel = Field(description="Details about the seniority and level of the position.")
    salary: Salary = Field(description="Salary information for the job.")
    metadata: Metadata = Field(description="Metadata associated with the job posting, like URLs and posting dates.")
    flexibleWorkArrangements: List[Any] = Field(default_factory=list, description="List of flexible work arrangements offered. Structure of items is undefined from sample.")
    score: float = Field(description="A relevance score assigned to this job result by the search algorithm.")
    postedCompany: PostedCompany = Field(description="Details of the company that posted the job listing.")
    employmentTypes: EmploymentType = Field(description="Type of employment offered (e.g., full-time, contract).")
    hiringCompany: Optional[Any] = Field(default=None, description="Details of the hiring company, if different from the posted company. Structure undefined from sample.")
    shiftPattern: Optional[Any] = Field(default=None, description="Details about the work shift pattern, if applicable. Structure undefined from sample.")
    categories: Category = Field(description="Primary category the job falls under.")

class JobSearchResponse(BaseModel):
    """Root model for the job search API response."""
    links: Links = Field(alias="_links", description="Navigation links for paginating through search results.")
    searchRankingId: str = Field(description="Identifier for the specific search ranking or session that produced these results.")
    results: JobResult = Field(description="The job result data. Based on the provided 'job_schema.json', this field contains a single job result object. In a typical list API, this might be a list of JobResult objects.")
    total: int = Field(description="Total number of job results matching the search criteria (potentially across all pages).")
    countWithoutFilters: int = Field(description="Total number of job results available before any search filters were applied.")

# Command-Line Interface (CLI) for testing

def main_cli():
    """Parses a JSON file using the defined Pydantic schema and prints the object."""
    parser = argparse.ArgumentParser(
        description="Validate and parse a job JSON file according to the MyCareersFuture schema."
    )
    parser.add_argument(
        "json_file", 
        type=str,
        help="Path to the JSON file to parse."
    )

    args = parser.parse_args()

    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            json_data_str = f.read()
        
        # Validate and parse the JSON data
        job_response = JobSearchResponse.model_validate_json(json_data_str)
        
        # Print the parsed Pydantic model
        # The default Pydantic __repr__ is quite informative.
        # For a more custom/pretty print, you might use job_response.model_dump_json(indent=2)
        print(job_response)

    except FileNotFoundError:
        print(f"Error: The file '{args.json_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{args.json_file}' contains invalid JSON.")
    except Exception as e:
        # This will catch Pydantic's ValidationError and other unexpected errors
        print(f"An error occurred while processing the file '{args.json_file}':")
        print(e)

if __name__ == "__main__":
    main_cli() 