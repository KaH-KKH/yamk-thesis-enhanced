*** Settings ***
Documentation    ** This test case verifies the functionality of the login user use case on the internet.herokuapp.com website.
Library          Browser
Test Tags        ** login user secure area
Library          OperatingSystem
Library          DateTime
Test Setup       Setup Browser
Test Teardown    Close Browser

*** Variables ***
${BASE_URL}      https://the-internet.herokuapp.com
${BROWSER}       chromium
${HEADLESS}      false
${TIMEOUT}       10s

*** Test Cases ***
**
    [Documentation]    ** This test case verifies the functionality of the login user use case on the internet.herokuapp.com website.
    [Tags]    ** login user secure area
    [Setup]    Run Keywords
    ...    Open the browser
    ...    Navigate to the login page at  the-internet.herokuapp.com/login
    
    Type Text    id="username"    test value
    Click    text="l"
    Log    Action: Validate credentials
    Log    Action: If credentials are valid, display a success message and allow access to the secure area
    Log    Action: If credentials are invalid, display an error message
    
    [Teardown]    Run Keywords
    ...    Close the browser
    ...    Expected Results:**
    ...    The user should be able to successfully log into the secure area with valid credentials.
    ...    The system should display a success message and allow the user to access the secure area if credentials are valid.
    ...    The system should display an error message if credentials are invalid.
    ...    ```

*** Keywords ***
Setup Browser
    New Browser    ${BROWSER}    headless=${HEADLESS}
    Set Browser Timeout    ${TIMEOUT}
    New Context    viewport={'width': 1280, 'height': 720}

Close Browser
    Take Screenshot    fullPage=True
    Close Browser    ALL

Login To Application
    [Arguments]    ${username}    ${password}
    Go To    ${BASE_URL}/login
    Type Text    id=username    ${username}
    Type Text    id=password    ${password}
    Click    css=button[type='submit']
    Wait For Elements State    css=.flash    visible    timeout=${TIMEOUT}
