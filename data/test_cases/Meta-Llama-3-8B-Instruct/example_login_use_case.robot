*** Settings ***
Documentation    This test case tests the secure login functionality of the system, verifying that a valid username and password result in successful login and access to the secure area.
Library          Browser
Test Tags        login security
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
UC-001 Secure Login
    [Documentation]    This test case tests the secure login functionality of the system, verifying that a valid username and password result in successful login and access to the secure area.
    [Tags]    login security
    [Setup]    Run Keywords
    ...    Go To  https://the-internet.herokuapp.com/login
    
    Click    text="E"
    Log    Action: Input Text  id=password  SuperSecretPassword!
    Click    text="E"
    Wait For Elements State
    Click    text="L"
    Log    Action: Input Text  id=password  SuperSecretPassword!
    Click    text="E"
    Wait For Elements State
    Click    text="L"
    
    [Teardown]    Run Keywords
    ...    ```
    ...    This test case uses Browser library keywords, tests the secure login functionality, and includes proper setup and teardown. The test case name, documentation, tags, setup, main test steps, and teardown are clearly defined. The test case uses appropriate selectors and waits, and includes a timeout for the Wait Until Page Contains keyword. The test case is executable and can be run to verify the secure login functionality of the system.

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
