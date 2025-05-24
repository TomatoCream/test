from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, date


class Skills(BaseModel):
    """Represents skills information for a job."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    uuid: str = Field(description="Unique identifier for the skill")
    confidence: Optional[float] = Field(default=None, description="Confidence score for skill matching")
    skill: str = Field(description="Name of the skill")


class EmploymentTypes(BaseModel):
    """Represents employment type information."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    id: int = Field(description="Employment type ID")
    employment_type: str = Field(alias="employmentType", description="Type of employment")


class Districts(BaseModel):
    """Represents district information for job location."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    id: int = Field(description="District ID")
    sectors: List[str] = Field(default_factory=list, description="List of sectors in the district")
    region_id: str = Field(alias="regionId", description="Region identifier")
    location: str = Field(description="Location name")
    region: str = Field(description="Region name")


class Address(BaseModel):
    """Represents address information for a job."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    unit: Optional[str] = Field(default=None, description="Unit number")
    foreign_address2: Optional[str] = Field(default=None, alias="foreignAddress2", description="Foreign address line 2")
    overseas_country: Optional[str] = Field(default=None, alias="overseasCountry", description="Overseas country name")
    street: Optional[str] = Field(default=None, description="Street name")
    lng: Optional[float] = Field(default=None, description="Longitude coordinate")
    lat: Optional[float] = Field(default=None, description="Latitude coordinate")
    is_overseas: Optional[bool] = Field(default=None, alias="isOverseas", description="Whether the job is overseas")
    foreign_address1: Optional[str] = Field(default=None, alias="foreignAddress1", description="Foreign address line 1")
    postal_code: Optional[str] = Field(default=None, alias="postalCode", description="Postal code")
    districts: Optional[List[Districts]] = Field(default=None, description="List of district information")
    block: Optional[str] = Field(default=None, description="Block number")
    floor: Optional[str] = Field(default=None, description="Floor number")
    building: Optional[str] = Field(default=None, description="Building name")


class Scheme(BaseModel):
    """Represents a scheme entity."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    id: int = Field(description="Scheme ID")
    scheme: str = Field(description="Scheme name")


class Schemes(BaseModel):
    """Represents scheme information for a job."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    expiry_date: Optional[date] = Field(default=None, alias="expiryDate", description="Scheme expiry date")
    start_date: Optional[date] = Field(default=None, alias="startDate", description="Scheme start date")
    scheme: Optional[Scheme] = Field(default=None, description="Scheme details")
    # sub_scheme: Optional[Scheme] = Field(default=None, alias="subScheme", description="Sub-scheme information")


class ResponsiveEmployer(BaseModel):
    """Represents responsive employer information."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    is_responsive: bool = Field(alias="isResponsive", description="Whether the employer is responsive")


class Company(BaseModel):
    """Represents a company entity from the MyCareersFuture API."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    ssic_code: Optional[str] = Field(default=None, alias="ssicCode", description="Singapore Standard Industrial Classification code")
    last_sync_date: datetime = Field(alias="lastSyncDate", description="Last synchronization date")
    employee_count: Optional[int] = Field(default=None, alias="employeeCount", description="Number of employees in the company")
    uen: str = Field(description="Unique Entity Number of the company")
    logo_upload_path: Optional[str] = Field(
        default=None, 
        alias="logoUploadPath", 
        description="URL path to the company logo"
    )
    responsive_employer: Optional[ResponsiveEmployer] = Field(
        default=None,
        alias="responsiveEmployer",
        description="Responsive employer information"
    )
    logo_file_name: Optional[str] = Field(
        default=None, 
        alias="logoFileName", 
        description="Filename of the company logo"
    )
    company_url: Optional[str] = Field(
        default=None, 
        alias="companyUrl", 
        description="Company website URL"
    )
    name: str = Field(description="Company name")
    ssic_code_2020: Optional[str] = Field(
        default=None, 
        alias="ssicCode2020", 
        description="Singapore Standard Industrial Classification code (2020 version)"
    )
    description: Optional[str] = Field(
        default=None, 
        description="Company description (may contain HTML)"
    )


class SalaryType(BaseModel):
    """Represents salary type information."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    id: int = Field(description="Salary type ID")
    salary_type: str = Field(alias="salaryType", description="Type of salary")


class Salary(BaseModel):
    """Represents salary information for a job."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    minimum: Optional[int] = Field(default=None, description="Minimum salary")
    type: SalaryType = Field(description="Salary type information")
    maximum: Optional[int] = Field(default=None, description="Maximum salary")


class PositionLevels(BaseModel):
    """Represents position level information."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    id: int = Field(description="Position level ID")
    position: str = Field(description="Position level name")


class Status(BaseModel):
    """Represents job status information."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    job_status: str = Field(alias="jobStatus", description="Current job status")
    id: int = Field(description="Status ID")


class Metadata(BaseModel):
    """Represents metadata information for a job."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    created_at: datetime = Field(alias="createdAt", description="Job creation timestamp")
    edit_count: int = Field(alias="editCount", description="Number of edits made to the job")
    email_recipient: Optional[str] = Field(default=None, alias="emailRecipient", description="Email recipient identifier")
    is_hide_salary: bool = Field(alias="isHideSalary", description="Whether salary is hidden")
    is_hide_company_address: bool = Field(alias="isHideCompanyAddress", description="Whether company address is hidden")
    repost_count: int = Field(alias="repostCount", description="Number of times job was reposted")
    is_hide_employer_name: bool = Field(alias="isHideEmployerName", description="Whether employer name is hidden")
    new_posting_date: date = Field(alias="newPostingDate", description="New posting date")
    job_details_url: str = Field(alias="jobDetailsUrl", description="URL to job details page")
    job_post_id: str = Field(alias="jobPostId", description="Job post identifier")
    deleted_at: Optional[datetime] = Field(default=None, alias="deletedAt", description="Job deletion timestamp")
    is_posted_on_behalf: bool = Field(alias="isPostedOnBehalf", description="Whether job was posted on behalf")
    original_posting_date: date = Field(alias="originalPostingDate", description="Original posting date")
    total_number_job_application: int = Field(alias="totalNumberJobApplication", description="Total number of applications")
    updated_at: datetime = Field(alias="updatedAt", description="Last update timestamp")
    is_hide_hiring_employer_name: bool = Field(alias="isHideHiringEmployerName", description="Whether hiring employer name is hidden")
    created_by: str = Field(alias="createdBy", description="Creator identifier")
    expiry_date: date = Field(alias="expiryDate", description="Job expiry date")
    matched_skills_score: Optional[float] = Field(default=None, alias="matchedSkillsScore", description="Matched skills score")
    total_number_of_view: int = Field(alias="totalNumberOfView", description="Total number of views")


class FlexibleWorkArrangements(BaseModel):
    """Represents flexible work arrangement information."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    id: int = Field(description="Flexible work arrangement ID")
    flexible_work_arrangement: str = Field(alias="flexibleWorkArrangement", description="Type of flexible work arrangement")


class Categories(BaseModel):
    """Represents job category information."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    id: int = Field(description="Category ID")
    category: str = Field(description="Category name")


class Job(BaseModel):
    """Represents a job listing."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"
    )
    
    psd_url: Optional[str] = Field(default=None, alias="psdUrl", description="PSD URL")
    skills: Optional[List[Skills]] = Field(default=None, description="List of skills information")
    employment_types: Optional[List[EmploymentTypes]] = Field(default=None, alias="employmentTypes", description="List of employment types")
    address: Optional[Address] = Field(default=None, description="Job address")
    ssoc_code: Optional[str] = Field(default=None, alias="ssocCode", description="Singapore Standard Occupational Classification code")
    minimum_years_experience: Optional[int] = Field(default=None, alias="minimumYearsExperience", description="Minimum years of experience required")
    job_titles: List[str] = Field(default_factory=list, alias="jobTitles", description="List of job titles")
    source_code: Optional[str] = Field(default=None, alias="sourceCode", description="Source code")
    schemes: Optional[List[Schemes]] = Field(default=None, description="List of scheme information")
    occupation_id: Optional[str] = Field(default=None, alias="occupationId", description="Occupation identifier")
    working_hours: Optional[str] = Field(default=None, alias="workingHours", description="Working hours")
    shift_pattern: Optional[str] = Field(default=None, alias="shiftPattern", description="Shift pattern")
    ssec_fos: Optional[str] = Field(default=None, alias="ssecFos", description="SSEC Field of Study")
    title: str = Field(description="Job title")
    hiring_company: Optional[Company] = Field(default=None, alias="hiringCompany", description="Hiring company information")
    salary: Optional[Salary] = Field(default=None, description="Salary information")
    position_levels: Optional[List[PositionLevels]] = Field(default=None, alias="positionLevels", description="List of position levels")
    uuid: str = Field(description="Unique identifier for the job")
    status: Optional[Status] = Field(default=None, description="Job status")
    number_of_vacancies: Optional[int] = Field(default=None, alias="numberOfVacancies", description="Number of vacancies")
    posted_company: Optional[Company] = Field(default=None, alias="postedCompany", description="Company that posted the job")
    ssec_eqa: Optional[str] = Field(default=None, alias="ssecEqa", description="SSEC Educational Qualification Attainment")
    metadata: Optional[Metadata] = Field(default=None, description="Job metadata")
    ssoc_version: Optional[str] = Field(default=None, alias="ssocVersion", description="SSOC version")
    flexible_work_arrangements: Optional[List[FlexibleWorkArrangements]] = Field(
        default=None, 
        alias="flexibleWorkArrangements", 
        description="List of flexible work arrangements"
    )
    categories: Optional[List[Categories]] = Field(default=None, description="List of job categories")
    description: Optional[str] = Field(default=None, description="Job description (may contain HTML)")
    other_requirements: Optional[str] = Field(default=None, alias="otherRequirements", description="Other requirements")


class CombinedResultsContainer(BaseModel):
    """Root model for the combined results container."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    results: List[Company] = Field(description="List of company results")


class JobResultsContainer(BaseModel):
    """Root model for the job results container."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    results: List[Job] = Field(description="List of job results")


# Example usage and validation
if __name__ == "__main__":
    import json
    
    # Sample company data
    company_sample_data = {
        "results": [
            {
                "ssicCode": "47721",
                "lastSyncDate": "2024-07-03T06:35:29.000Z",
                "employeeCount": 111212223,
                "uen": "180000001W",
                "logoUploadPath": "https://static.mycareersfuture.gov.sg/images/company/logos/588964938e9582c6cc9676c763a7ae6b/COMMUNICABLE%20DISEASE%20THREATS%20INITIATIVE,%20INCORPORATING%20ASIA%20PACIFIC%20LEADERS%20MALARIA%20ALLIANCE.png",
                "logoFileName": "588964938e9582c6cc9676c763a7ae6b/COMMUNICABLE DISEASE THREATS INITIATIVE, INCORPORATING ASIA PACIFIC LEADERS MALARIA ALLIANCE.png",
                "companyUrl": "https://nam12.safelinks.protection.outlook.com/?url=http%3A%2F%2Fwww.haoyangtechnology.sg%2F&data=05%7C01%7CChen.Jian%40bakerhughes.com%7C752e7c6e6b1a4e519ddf08db1542afe6%7Cd584a4b7b1f24714a578fd4d43c146a6%7C0%7C0%7C638127148400348541%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=9akHzTkHZHFXIwtQVxPIOXGvDWrWHm9Un7Mv4V4vxps%3D&reserved=0",
                "name": "Persatuan Ulama dan Guru-Guru Agama Islam (Singapura) (Singapore Islamic Scholars and Religious Teac",
                "ssicCode2020": "47721",
                "description": "<p>UNIQUANT FUND MANAGEMENT PTE. LTD.</p>    "
            }
        ]
    }
    
    # Sample job data (simplified without links)
    job_sample_data = {
        "results": [
            {
                "psdUrl": None,
                "skills": {
                    "uuid": "1de5bf4a69572155cb8c297ac21a161d",
                    "confidence": None,
                    "skill": "Ability To Work Independently"
                },
                "employmentTypes": {
                    "id": 8,
                    "employmentType": "Full Time"
                },
                "address": {
                    "unit": None,
                    "foreignAddress2": " Johor Darul Ta'zim, Malaysia",
                    "overseasCountry": "United Arab Emirates",
                    "street": "ANG MO KIO ELECTRONICS PARK ROAD",
                    "lng": 104.015422009394,
                    "lat": 1.46980033619074,
                    "isOverseas": True,
                    "foreignAddress1": "Rama 9,  Unilever House, Huaikwhang  Unilever House  Bangkok",
                    "postalCode": "188778",
                    "districts": {
                        "id": 998,
                        "sectors": [],
                        "regionId": "Islandwide",
                        "location": "Islandwide",
                        "region": "Islandwide"
                    },
                    "block": "Wintech Ct",
                    "floor": None,
                    "building": "NATIONAL UNIVERSITY HOSPITAL (NATIONAL UNIVERSITY HLTH SYSTM BLDG"
                },
                "ssocCode": "35121",
                "minimumYearsExperience": 30,
                "jobTitles": [],
                "sourceCode": "Employer Portal",
                "schemes": {
                    "expiryDate": "2052-03-10",
                    "startDate": "2022-03-10",
                    "scheme": {
                        "id": 7,
                        "scheme": "Mid-Career Pathways Programme"
                    },
                    "subScheme": None
                },
                "occupationId": "OCC060957",
                "workingHours": None,
                "shiftPattern": None,
                "ssecFos": "0919",
                "title": "Senior/ Sales & Marketing Executive (Telecommunication)",
                "hiringCompany": {
                    "ssicCode": "81211",
                    "lastSyncDate": "2025-05-24T16:10:14.000Z",
                    "employeeCount": 27000,
                    "uen": "202409101R",
                    "logoUploadPath": "https://static.mycareersfuture.gov.sg/images/company/logos/8f9d97f4826142cbadd04b254a8b806f/china-railway-engineering-equipment-group-co-singapore-branch.jpg",
                    "logoFileName": "8f9d97f4826142cbadd04b254a8b806f/china-railway-engineering-equipment-group-co-singapore-branch.jpg",
                    "companyUrl": "https://nam12.safelinks.protection.outlook.com/?url=http%3A%2F%2Fwww.haoyangtechnology.sg%2F",
                    "name": "CHINA RAILWAY ENGINEERING EQUIPMENT GROUP CO., LTD. Singapore Branch",
                    "ssicCode2020": "43301",
                    "description": "<p>At Alstern Technologies, we don't just solve engineering challenges—we conquer them.</p>"
                },
                "salary": {
                    "minimum": 200000,
                    "type": {
                        "id": 4,
                        "salaryType": "Monthly"
                    },
                    "maximum": 243000
                },
                "positionLevels": {
                    "id": 12,
                    "position": "Fresh/entry level"
                },
                "uuid": "22ee69c711f06dd0736e37ba869a61c9",
                "status": {
                    "jobStatus": "Re-open",
                    "id": 103
                },
                "numberOfVacancies": 999,
                "postedCompany": {
                    "ssicCode": "78104",
                    "lastSyncDate": "2025-05-24T16:15:37.000Z",
                    "employeeCount": 600000,
                    "uen": "200803548G",
                    "logoUploadPath": "https://static.mycareersfuture.gov.sg/images/company/logos/07b893acd1b19d6bdb96ac7a32df70a6/PRINCETON%20DIGITAL%20GROUP%20(SINGAPORE)%20MANAGEMENT%20PRIVATE%20LIMITED.jpg",
                    "responsiveEmployer": {
                        "isResponsive": True
                    },
                    "logoFileName": "fb112cf3b074dd181d01d8ef33dae41e/china-energy-engineering-group-shanxi-no3-electric-power-construction-co.jpg",
                    "companyUrl": "https://www.hiexpress.com/hotels/gb/en/singapore/sincq/hoteldetail",
                    "name": "CHINA ENERGY ENGINEERING GROUP SHANXI NO.3 ELECTRIC POWER CONSTRUCTION CO., LTD. (Singapore Branch)",
                    "ssicCode2020": "78104",
                    "description": "<h2>Company Overview</h2>\n<p><strong>If you're seeking for employment or employees, you're at the right place.</strong></p>"
                },
                "ssecEqa": "43",
                "metadata": {
                    "createdAt": "2025-05-24T17:53:53.000Z",
                    "editCount": 2,
                    "emailRecipient": "ba5af493-7139-4c4e-a997-cc9cbfb623a1",
                    "isHideSalary": False,
                    "isHideCompanyAddress": False,
                    "repostCount": 2,
                    "isHideEmployerName": False,
                    "newPostingDate": "2025-05-25",
                    "jobDetailsUrl": "https://www.mycareersfuture.gov.sg/job/education-training/job-title-lecturer",
                    "jobPostId": "MCF-2025-0752373",
                    "deletedAt": "2025-05-21T17:15:04.000Z",
                    "isPostedOnBehalf": True,
                    "originalPostingDate": "2025-05-25",
                    "totalNumberJobApplication": 1071,
                    "updatedAt": "2025-05-24T17:53:53.000Z",
                    "isHideHiringEmployerName": False,
                    "createdBy": "ba5af493-7139-4c4e-a997-cc9cbfb623a1",
                    "expiryDate": "2025-06-24",
                    "matchedSkillsScore": None,
                    "totalNumberOfView": 57502
                },
                "ssocVersion": "2020v2",
                "flexibleWorkArrangements": {
                    "id": 5,
                    "flexibleWorkArrangement": "Compressed Work Schedule"
                },
                "categories": {
                    "id": 35,
                    "category": "Information Technology"
                },
                "description": "<h2><strong>Fast-Track A Teaching Career &amp; Be Rewarded Greatly as a Language Specialist with EduEdge!</strong></h2>",
                "otherRequirements": None
            }
        ]
    }
    
    # Validate the schemas
    try:
        # Test company schema
        company_response = CombinedResultsContainer(**company_sample_data)
        print("✅ Company schema validation successful!")
        print(f"Number of companies: {len(company_response.results)}")
        print(f"First company: {company_response.results[0].name}")
        
        # Test job schema
        job_response = JobResultsContainer(**job_sample_data)
        print("✅ Job schema validation successful!")
        print(f"Number of jobs: {len(job_response.results)}")
        if job_response.results:
            job = job_response.results[0]
            print(f"First job title: {job.title}")
            print(f"Job UUID: {job.uuid}")
            if job.hiring_company:
                print(f"Hiring company: {job.hiring_company.name}")
            if job.salary:
                print(f"Salary range: {job.salary.minimum} - {job.salary.maximum}")
    except Exception as e:
        print(f"❌ Schema validation failed: {e}") 