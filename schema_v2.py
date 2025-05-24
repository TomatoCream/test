from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime


class Company(BaseModel):
    """Represents a company entity from the MyCareersFuture API."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    ssic_code: str = Field(alias="ssicCode", description="Singapore Standard Industrial Classification code")
    last_sync_date: datetime = Field(alias="lastSyncDate", description="Last synchronization date")
    employee_count: int = Field(alias="employeeCount", description="Number of employees in the company")
    uen: str = Field(description="Unique Entity Number of the company")
    logo_upload_path: Optional[str] = Field(
        default=None, 
        alias="logoUploadPath", 
        description="URL path to the company logo"
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


class CompanySearchResponse(BaseModel):
    """Root model for the company search API response."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    results: List[Company] = Field(description="List of company results")


# Example usage and validation
if __name__ == "__main__":
    import json
    
    # Sample data based on the provided JSON
    sample_data = {
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
    
    # Validate the schema
    try:
        response = CompanySearchResponse(**sample_data)
        print("✅ Schema validation successful!")
        print(f"Number of companies: {len(response.results)}")
        print(f"First company: {response.results[0].name}")
        print(f"UEN: {response.results[0].uen}")
        print(f"Employee count: {response.results[0].employee_count}")
    except Exception as e:
        print(f"❌ Schema validation failed: {e}") 