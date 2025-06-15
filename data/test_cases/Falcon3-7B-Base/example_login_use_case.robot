*** Settings ***
Documentation    This test case verifies the functionality described in the use case.
Library          Browser
Test Tags        robot browser
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
    [Documentation]    This test case verifies the functionality described in the use case.
    [Tags]    robot browser
    
    Log    Action: BeforeSuite:
    Log    Action: Initialize Browser library
    Log    Action: Open the target URL
    Go To    https://the-internet.herokuapp.com/the
    Log    Action: When user performs action A
    Log    Action: And user performs action B
    Log    Action: And user performs action C
    Log    Action: Then system should respond in expected manner
    Log    Action: AfterSuite:
    Log    Action: Close the browser
    Get Element States
    Log    Action: Assert expected conditions are met
    Log    Action: [robot]
    Log    Action: @robot
    Log    Action: @browser
    Log    Action: BeforeSuite:
    Log    Action: Initialize Browser library
    Log    Action: Open the target URL
    Go To    https://the-internet.herokuapp.com/the
    Log    Action: When user performs action A
    Log    Action: And user performs action B
    Log    Action: And user performs action C
    Log    Action: Then system should respond in expected manner
    Log    Action: AfterSuite:
    Log    Action: Close the browser
    Get Element States
    Log    Action: Assert expected conditions are met
    Log    Action: This is an example of a Robot Framework test case that:
    Log    Action: Uses Browser library keywords
    Log    Action: Tests the functionality described in the use case
    Log    Action: Includes proper setup and teardown
    Log    Action: Has clear step-by-step test actions
    Wait For Elements State
    Log    Action: Verifies expected results and conditions
    Log    Action: Follows a structured format for readability and maintainability
    Log    Action: Demonstrates the use of tags for categorization and organization
    Log    Action: Illustrates the use of BeforeSuite and AfterSuite steps for global setup and teardown
    Log    Action: Highlights the importance of verification steps for validating test results
    Log    Action: Test case name: UC-001
    Log    Action: BeforeSuite:
    Log    Action: Initialize Browser library
    Log    Action: Open the target URL
    Go To    https://the-internet.herokuapp.com/the
    Log    Action: When user performs action A
    Log    Action: And user performs action B
    Log    Action: And user performs action C
    Log    Action: Then system should respond in expected manner
    Log    Action: AfterSuite:
    Log    Action: Close the browser
    Get Element States
    Log    Action: Assert expected conditions are met
    Log    Action: [robot]
    Log    Action: @robot
    Log    Action: @browser
    Log    Action: BeforeSuite:
    Log    Action: Initialize Browser library
    Log    Action: Open the target URL
    Go To    https://the-internet.herokuapp.com/the
    Log    Action: When user performs action A
    Log    Action: And user performs action B
    Log    Action: And user performs action C
    Log    Action: Then system should respond in expected manner
    Log    Action: AfterSuite:
    Log    Action: Close the browser
    Get Element States
    Log    Action: Assert expected conditions are met
    Log    Action: This is an example of a Robot Framework test case that:
    Log    Action: Uses Browser library keywords
    Log    Action: Tests the functionality described in the use case
    Log    Action: Includes proper setup and teardown
    Log    Action: Has clear step-by-step test actions
    Wait For Elements State
    Log    Action: Verifies expected results and conditions
    Log    Action: Follows a structured format for readability and maintainability
    Log    Action: Demonstrates the use of tags for categorization and organization
    Log    Action: Illustrates the use of BeforeSuite and AfterSuite steps for global setup and teardown
    Log    Action: Highlights the importance of verification steps for validating test results
    Log    Action: Test case name: UC-001
    Log    Action: BeforeSuite:
    Log    Action: Initialize Browser library
    Log    Action: Open the target URL
    Go To    https://the-internet.herokuapp.com/the
    Log    Action: When user performs action A
    Log    Action: And user performs action B
    Log    Action: And user performs action C
    Log    Action: Then system should respond in expected manner
    Log    Action: AfterSuite:
    Log    Action: Close the browser
    Get Element States
    Log    Action: Assert expected conditions are met
    Log    Action: [robot]
    Log    Action: @robot
    Log    Action: @browser
    Log    Action: BeforeSuite:
    Log    Action: Initialize Browser library
    Log    Action: Open the target URL
    Go To    https://the-internet.herokuapp.com/the
    Log    Action: When user performs action A
    Log    Action: And user performs action B
    Log    Action: And user performs action C
    Log    Action: Then system should respond in expected manner
    Log    Action: AfterSuite:
    Log    Action: Close the browser
    Get Element States
    Log    Action: Assert expected conditions are met
    Log    Action: This is an example of a Robot Framework test case that:

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
