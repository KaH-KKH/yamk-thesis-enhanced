*** Settings ***
Documentation    ** This test case verifies the functionality of the login process to the secure area on the internet.herokuapp.com website.
Library          Browser
Test Tags        ** login secure area user authentication
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
    [Documentation]    ** This test case verifies the functionality of the login process to the secure area on the internet.herokuapp.com website.
    [Tags]    ** login secure area user authentication
    [Setup]    Run Keywords
    ...    Open the browser
    ...    Navigate to the login page at  https://the-internet.herokuapp.com/login
    
    Type Text    id="valid"    test value
    Click    text="""
    Get Text
    Get Text
    Log    Action: **Alternative Flows:**
    Type Text    id="field"    test value
    Click    text="""
    
    [Teardown]    Run Keywords
    ...    Close the browser
    ...    Expected Result:**
    ...    The user is able to successfully login to the secure area with valid credentials and access the features and functionality of the secure area. Alternatively, if the user enters invalid credentials or forgets their password, appropriate error messages are displayed.

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
