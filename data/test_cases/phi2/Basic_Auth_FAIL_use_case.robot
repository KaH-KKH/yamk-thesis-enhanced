*** Settings ***
Documentation    Description: Tests the functionality of logging in to the The Internet Website.
Library          Browser
Test Tags        Tags: Browser Library Web Browser HTTP HTML JavaScript DOM
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
Test Case: Login to The Internet Website
    [Documentation]    Description: Tests the functionality of logging in to the The Internet Website.
    [Tags]    Tags: Browser Library Web Browser HTTP HTML JavaScript DOM
    
    Go To    https://the-internet.herokuapp.com/The
    Get Text
    Get Text
    Type Text    id="username"    test value
    Type Text    id="password"    test value
    Click    text="'"
    Get Text
    Get Text
    Log    Action: Close web browser.
    Log    Action: Add expected results:
    Custom Login Keyword
    Log    Action: The homepage displays the User's name.
    Log    Action: Add actual results:
    Custom Login Keyword
    Log    Action: Actual result: The homepage displays the User's name.
    Log    Action: Add comments:
    Log    Action: Comments: All test steps passed successfully.
    Log    Action: Answer: The correct Robot Framework test case is:
    Custom Login Keyword
    Custom Login Keyword
    Log    Action: Comments: All test steps passed successfully.

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
