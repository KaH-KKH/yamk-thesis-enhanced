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
    ...    Open Chrome browser
    ...    Navigate to the login page at the-internet.herokuapp.com/login
    
    Type Text    id="username"    john.doe@example.com
    Type Text    id="password"    secret123
    Click    text="""
    Get Text
    Get Text
    
    [Teardown]    Run Keywords
    ...    Close Chrome browser
    ...    Expected Result:**
    ...    User is successfully logged into the secure area
    ...    User can access the features and functionality of the secure area

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
