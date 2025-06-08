*** Settings ***
Documentation    ** This test case verifies the functionality of logging into the secure area of the internet.herokuapp.com website.
Library          Browser
Test Tags        ** login secure area the-internet.herokuapp.com
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
    [Documentation]    ** This test case verifies the functionality of logging into the secure area of the internet.herokuapp.com website.
    [Tags]    ** login secure area the-internet.herokuapp.com
    [Setup]    Run Keywords
    ...    Open the browser
    ...    Navigate to the login page at https://the-internet.herokuapp.com/login
    
    Type Text    id="username"    test value
    Click    text="""
    Log    Action: Validate credentials
    Log    Action: If credentials are valid, display a success message and allow user to access the secure area
    Log    Action: If credentials are invalid, display an error message
    
    [Teardown]    Run Keywords
    ...    Close the browser
    ...    Alternative Flows:**
    ...    If the user forgets their password, click on "Forgot Password" to reset their password
    ...    If the user's account is locked due to too many failed login attempts, wait for a certain amount of time before trying to log in again
    ...    Post-Conditions:**
    ...    User is logged into the secure area
    ...    User can access the features and functionality of the secure area
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
