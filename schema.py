from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any, Union
from datetime import date, datetime
import argparse
import json

# Pydantic Model Definitions

class Link(BaseModel):
    """Represents a hyperlink, typically for navigation."""
    href: str = Field(description="URL for the link.")

class Links(BaseModel):
    """Container for navigation links, commonly used in paginated API responses."""
    next: Optional[Link] = Field(default=None, description="Link to the next page of results.")
    self: Link = Field(description="Link to the current page of results.")
    first: Optional[Link] = Field(default=None, description="Link to the first page of results.")
    last: Optional[Link] = Field(default=None, description="Link to the last page of results.")

class Districts(BaseModel):
    """Detailed district information for an address."""
    sectors: List[str] = Field(description="Sector identifier for the district.")
    location: str = Field(description="Descriptive name of the location or broader area.")
    region: str = Field(description="Geographical region (e.g., 'Islandwide').")
    id: int = Field(description="Unique identifier for the district.")
    regionId: str = Field(description="Identifier for the region, often mirroring the region name.")

class Address(BaseModel):
    """Represents a physical address."""
    floor: Optional[str] = Field(default=None, description="Floor number of the address, if applicable.")
    postalCode: Optional[str] = Field(default=None, description="Postal code of the address.")
    districts: List[Districts] = Field(description="Detailed district information.")
    isOverseas: bool = Field(description="Flag indicating if the address is located overseas.")
    foreignAddress2: Optional[str] = Field(default=None, description="Second line of foreign address, if applicable.")
    street: Optional[str] = Field(default=None, description="Street name and number.")
    overseasCountry: Optional[str] = Field(default=None, description="Country name, if the address is overseas.")
    block: Optional[str] = Field(default=None, description="Block number of the address. E.g., '391B'") # Sample had "391B", making it Optional for wider applicability
    lat: Optional[float] = Field(default=None, description="Latitude coordinate of the address.")
    lng: Optional[float] = Field(default=None, description="Longitude coordinate of the address.")
    unit: Optional[str] = Field(default=None, description="Unit number of the address, if applicable.")
    building: Optional[str] = Field(default=None, description="Name of the building. E.g., 'MARINA BAY FINANCIAL CENTRE'") # Sample had a value, making it Optional for wider applicability
    foreignAddress1: Optional[str] = Field(default=None, description="First line of foreign address, if applicable.")

class Skill(BaseModel):
    """Represents a skill required or associated with a job."""
    uuid: str = Field(description="Unique identifier for the skill.")
    skill: str = Field(description="Name of the skill (e.g., 'Oral & Written Communication Skills').")
    confidence: Optional[Any] = Field(default=None, description="Confidence score for the skill, if available.")

class Status(BaseModel):
    """Represents the current status of a job posting."""
    id: Union[str, int] = Field(description="Identifier for the job status (e.g., '102' or 102).")
    jobStatus: str = Field(description="Descriptive status of the job posting (e.g., 'Re-open').")

class PositionLevel(BaseModel):
    """Represents the seniority or level of a job position."""
    position: str = Field(description="Description of the position level (e.g., 'Senior Management').")
    id: int = Field(description="Identifier for the position level.")

class SalaryType(BaseModel):
    """Specifies the type or frequency of salary payment."""
    id: Optional[int] = Field(default=None, description="Identifier for the salary type, if available.")
    salaryType: str = Field(description="Type of salary payment (e.g., 'Monthly').")

class Salary(BaseModel):
    """Represents salary information for a job."""
    maximum: int = Field(description="Maximum salary amount for the position.")
    minimum: int = Field(description="Minimum salary amount for the position.")
    type: SalaryType = Field(description="Details about the salary payment type.")

class Metadata(BaseModel):
    """Contains metadata related to a job posting."""
    jobDetailsUrl: str = Field(description="URL to the full job details page on the job portal.")
    isPostedOnBehalf: bool = Field(description="Flag indicating if the job is posted on behalf of another company.")
    isHideHiringEmployerName: bool = Field(description="Flag indicating if the hiring employer's name is hidden from applicants.")
    jobPostId: str = Field(description="Unique identifier for the job post (e.g., 'MCF-2025-0714896').")
    totalNumberJobApplication: int = Field(description="Total number of applications received for this job posting.")
    isHideSalary: bool = Field(description="Flag indicating if the salary information is hidden from applicants.")
    newPostingDate: date = Field(description="Date when the job was newly posted or most recently reposted.")
    updatedAt: datetime = Field(description="Timestamp of the last update to the job posting information.")
    deletedAt: Optional[datetime] = Field(default=None, description="Timestamp when the job post was soft deleted.")
    createdBy: Optional[str] = Field(default=None, description="Identifier of the user/system that created the job post.")
    createdAt: Optional[datetime] = Field(default=None, description="Timestamp of when the job post was created.")
    emailRecipient: Optional[str] = Field(default=None, description="Identifier of the email recipient related to the job post.")
    editCount: Optional[int] = Field(default=None, description="Number of times the job post has been edited.")
    repostCount: Optional[int] = Field(default=None, description="Number of times the job post has been reposted.")
    totalNumberOfView: Optional[int] = Field(default=None, description="Total number of views for the job post.")
    originalPostingDate: Optional[date] = Field(default=None, description="The very first date the job was posted.")
    expiryDate: Optional[date] = Field(default=None, description="Date when the job posting will expire.")
    isHideCompanyAddress: Optional[bool] = Field(default=None, description="Flag indicating if the company address is hidden from applicants.")
    isHideEmployerName: Optional[bool] = Field(default=None, description="Flag indicating if the employer's name (distinct from hiring employer) is hidden.")

class ResponsiveEmployer(BaseModel):
    """Indicates whether an employer is considered responsive."""
    isResponsive: bool = Field(description="Flag indicating if the employer is responsive to applications.")

class PostedCompanyLinks(BaseModel):
    """Navigation links related to a posted company."""
    self: Link = Field(description="Link to the company's own details.")
    jobs: Link = Field(description="Link to the jobs posted by this company.")
    addresses: Link = Field(description="Link to the company's addresses.")
    schemes: Link = Field(description="Link to schemes related to this company.")

class PostedCompany(BaseModel):
    """Details of the company that posted the job."""
    # The '_links' field from the source JSON is intentionally ignored in this model.
    uen: str = Field(description="Unique Entity Number (UEN) of the company.")
    logoUploadPath: Optional[str] = Field(default=None, description="Path or URL to the company's logo image.")
    responsiveEmployer: Optional[ResponsiveEmployer] = Field(default=None, description="Information about the employer's responsiveness.")
    logoFileName: Optional[str] = Field(default=None, description="Filename of the company's logo.")
    name: str = Field(description="Name of the company that posted the job.")
    description: Optional[str] = Field(default=None, description="Description of the company, may contain HTML.")
    ssicCode: Optional[str] = Field(default=None, description="SSIC (Singapore Standard Industrial Classification) code of the company.")
    employeeCount: Optional[int] = Field(default=None, description="Number of employees in the company.")
    companyUrl: Optional[str] = Field(default=None, description="URL to the company's website.")
    lastSyncDate: Optional[datetime] = Field(default=None, description="Timestamp of the last synchronization of company data.")
    ssicCode2020: Optional[str] = Field(default=None, description="SSIC code (2020 version) of the company.")
    badges: List[Any] = Field(default_factory=list, description="List of badges associated with the company.")

class JobEmploymentType(BaseModel):
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
    skills: List[Skill] = Field(description="Primary skill associated with the job. Based on the sample, this is a single skill object.")
    status: Status = Field(description="Current status of the job posting.")
    title: str = Field(description="Title of the job position.")
    positionLevels: List[PositionLevel] = Field(description="Details about the seniority and level of the position.")
    salary: Salary = Field(description="Salary information for the job.")
    metadata: Metadata = Field(description="Metadata associated with the job posting, like URLs and posting dates.")
    flexibleWorkArrangements: List[Any] = Field(default_factory=list, description="List of flexible work arrangements offered. Structure of items is undefined from sample.")
    score: Optional[float] = Field(default=None, description="A relevance score assigned to this job result by the search algorithm.")
    postedCompany: PostedCompany = Field(description="Details of the company that posted the job listing.")
    employmentTypes: List[JobEmploymentType] = Field(description="Type of employment offered (e.g., full-time, contract).")
    hiringCompany: Optional[Any] = Field(default=None, description="Details of the hiring company, if different from the posted company. Structure undefined from sample.")
    shiftPattern: Optional[Any] = Field(default=None, description="Details about the work shift pattern, if applicable. Structure undefined from sample.")
    categories: List[Category] = Field(description="Primary category the job falls under.")

class JobSearchResponse(BaseModel):
    """Root model for the job search API response."""
    # The '_links' field from the source JSON is intentionally ignored in this model.
    searchRankingId: Optional[str] = Field(default=None, description="Identifier for the specific search ranking or session that produced these results.")
    results: List[JobResult] = Field(description="The job result data. Based on the provided 'job_schema.json', this field contains a single job result object. In a typical list API, this might be a list of JobResult objects.")
    total: int = Field(description="Total number of job results matching the search criteria (potentially across all pages).")
    countWithoutFilters: int = Field(description="Total number of job results available before any search filters were applied.")

# Pydantic Model Definitions for Profile Data

class EmploymentType(BaseModel):
    """Represents the employment type in a user profile."""
    id: str = Field(description="Identifier for the employment type.")
    name: str = Field(description="Name of the employment type (e.g., 'Flexi Time').")

class Ssoc(BaseModel):
    """Represents an SSOC (Singapore Standard Occupational Classification) entry."""
    ssoc: str = Field(description="SSOC code.")
    ssocTitle: str = Field(description="Title or description of the SSOC code.")

class Ssic(BaseModel):
    """Represents an SSIC (Singapore Standard Industrial Classification) entry."""
    code: str = Field(description="SSIC code.")
    description: str = Field(description="Description of the SSIC code.")

class Country(BaseModel):
    """Represents a country entry."""
    codeNumber: str = Field(description="Numeric code for the country.")
    code: str = Field(description="Alpha code for the country (e.g., 'AF').")
    description: str = Field(description="Name of the country.")

class EmploymentStatus(BaseModel):
    """Represents the employment status in a user profile."""
    id: str = Field(description="Identifier for the employment status.")
    description: str = Field(description="Description of the employment status (e.g., 'Self-employed').")

class SsecEqa(BaseModel):
    """Represents an SSEC EQA (Educational Qualification Attainment) entry."""
    code: str = Field(description="SSEC EQA code.")
    description: str = Field(description="Description of the SSEC EQA.")

class SsecFos(BaseModel):
    """Represents an SSEC FOS (Field of Study) entry."""
    code: str = Field(description="SSEC FOS code.")
    description: str = Field(description="Description of the SSEC FOS.")

class CommonData(BaseModel):
    """Container for common profile-related data."""
    employmentTypes: List[EmploymentType] = Field(description="Details about employment type.")
    ssocList: List[Ssoc] = Field(description="SSOC information.")
    ssicList: List[Ssic] = Field(description="SSIC information.")
    countriesList: List[Country] = Field(description="Country information.")
    employmentStatusList: List[EmploymentStatus] = Field(description="Employment status information.")
    ssecEqaList: List[SsecEqa] = Field(description="SSEC EQA information.")
    ssecFosList: List[SsecFos] = Field(description="SSEC FOS information.")

class Data(BaseModel):
    """Container for the main profile data."""
    common: CommonData = Field(description="Common profile data.")

class Response(BaseModel):
    """Root model for the profile API response."""
    data: Data = Field(description="The profile data.")

# Custom JSON Encoder for Pydantic Types
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode='json')
        return super().default(obj)

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
        
        # Print the parsed Pydantic model as a JSON string using CustomEncoder
        job_response_json_str = job_response.model_dump_json(indent=2)
        print(job_response_json_str)

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
