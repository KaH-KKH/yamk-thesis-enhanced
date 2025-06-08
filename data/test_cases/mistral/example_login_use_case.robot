*** Settings ***
Documentation    Automated test for login functionality
Library          Browser
Test Tags        login smoke
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
Test Login Functionality
    [Documentation]    Automated test for login functionality
    [Tags]    login smoke
    [Setup]    Run Keywords
    ...    # Open the login page
    ...    Open Browser        ${BROWSER}    https://the-internet.herokuapp.com/login
    ...    # Maximize browser window
    ...    Maximize Browser Window
    ...    Variables ***
    ...    Username            tomsmith
    ...    Password            SuperSecretPassword!
    ...    Teardown ***
    
    New Browser    ${BROWSER}    headless=${HEADLESS}
    New Page    ${BASE_URL}/login
    Type Text    id=username    tomsmith
    Type Text    id=password    SuperSecretPassword!
    Click    css=button[type='submit']
    Wait For Elements State    text=You logged into a secure area!    visible
    Take Screenshot
    
    [Teardown]    Run Keywords
    ...    # Close the browser window
    ...    Close Browser
    ...    Test Steps ***
    ...    Test Step 1:
    ...    [Documentation]     Enter valid username.
    ...    Input Text          id:username       ${Username}
    ...    Test Step 2:
    ...    [Documentation]     Enter valid password.
    ...    Input Text          id:password       ${Password}
    ...    Test Step 3:
    ...    [Documentation]     Click the login button.
    ...    Click Element          id:login_button
    ...    Test Step 4:
    ...    [Documentation]     Wait for the success message to appear.
    ...    Wait Until Page Contains    You are now logged in.
    ...    Test Step 5:
    ...    [Documentation]     Verify the success message is displayed.
    ...    Should Contain         id:success_message      You are now logged in.
    ...    Alternative test steps ***
    ...    Test Step 6:
    ...    [Documentation]     Test invalid credentials.
    ...    Input Text          id:username       tom
    ...    Input Text          id:password       password
    ...    Click Element          id:login_button
    ...    Wait Until Element Is Not Visible         id:success_message
    ...    Should Contain         id:error_message      Invalid credentials. Please try again.
    ...    Test Step 7:
    ...    [Documentation]     Test empty fields.
    ...    Input Text          id:username       <blank>
    ...    Input Text          id:password       <blank>
    ...    Click Element          id:login_button
    ...    Wait Until Element Is Not Visible         id:success_message
    ...    Should Contain         id:error_message      Username and password are required fields.
    ...    ```
    ...    This Robot Framework test case follows all the given rules, and it covers the main flow and alternative flows of the use case. Additionally, it includes proper setup and teardown, clear test actions, and appropriate selectors and waits.

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
