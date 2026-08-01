"""The HR and Finance policy corpus.

Lifted out of `scripts/ingest.py` so both the vector ingester and the GraphRAG
indexer read one source of truth.

Two properties matter for GraphRAG and are deliberate, not incidental:

1. **Topics cross the department boundary.** Travel rules live in Finance but
   conference travel is HR; equipment budgets are Finance but remote-work
   equipment is HR; SOX controls are Finance but the conduct hotline is HR. If
   Leiden communities merely rediscovered the HR/Finance split, the graph would
   add nothing over the existing per-department indexes.
2. **Entities repeat across documents.** The same approvers (Manager, VP, CFO,
   Board), systems (Workday, Expensify, Coupa, Navan), and thresholds ($5,000,
   $25,000, $50,000, $100,000) recur, so extraction produces a connected graph
   rather than 75 disjoint islands.

HR-001..HR-008 and FIN-001..FIN-008 are unchanged from the original corpus so
the existing eval cases in `evals/questions.jsonl` stay valid.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document

HR_DOCS: list[Document] = [
    Document(
        page_content="""Paid Time Off (PTO) Policy. All full-time employees accrue 15 days of paid vacation per year during their first three years of employment. After three years, accrual increases to 20 days per year. PTO is accrued monthly at 1.25 days per month (or 1.67 days for tenured employees). Unused PTO can be carried over up to a maximum of 5 days into the next calendar year. Any excess is forfeited on January 1st. PTO requests must be submitted through Workday at least two weeks in advance for requests longer than 3 consecutive days.""",
        metadata={"source": "HR-001", "title": "PTO Policy", "department": "hr"},
    ),
    Document(
        page_content="""Parental Leave Policy. Birthing parents are eligible for 12 weeks of fully paid parental leave. Non-birthing parents (including adoptive and foster parents) receive 8 weeks of fully paid leave. Leave must be taken within 12 months of the child's birth or adoption. Employees must notify HR at least 30 days before intended leave start date when foreseeable. During parental leave, health insurance benefits continue without interruption and the employee's position is protected.""",
        metadata={"source": "HR-002", "title": "Parental Leave", "department": "hr"},
    ),
    Document(
        page_content="""Remote Work Policy. Employees may work remotely up to 3 days per week with manager approval. Fully remote arrangements require VP-level approval and are evaluated case-by-case based on role requirements. Remote workers must maintain core working hours of 10 AM to 3 PM in their assigned time zone for meeting availability. The company provides a $500 one-time home office stipend for fully remote employees to purchase equipment. Hybrid employees receive $250.""",
        metadata={"source": "HR-003", "title": "Remote Work", "department": "hr"},
    ),
    Document(
        page_content="""Performance Review Process. Formal performance reviews occur twice per year: mid-year in June and annual in December. Reviews use a 5-point scale: 1 (Below Expectations), 2 (Partially Meets), 3 (Meets Expectations), 4 (Exceeds), 5 (Outstanding). Employees complete a self-assessment two weeks before the review meeting. Merit increases and promotion decisions are based on the annual December review. Employees rated 1 or 2 are placed on a 90-day Performance Improvement Plan (PIP).""",
        metadata={"source": "HR-004", "title": "Performance Reviews", "department": "hr"},
    ),
    Document(
        page_content="""Onboarding Process for New Hires. New employees complete a 30-day onboarding program. Week 1 focuses on orientation, IT setup, and benefits enrollment — employees must enroll in benefits within 30 days of start date. Week 2 covers department-specific training. Weeks 3-4 involve shadowing and initial project assignment. New hires are assigned an onboarding buddy and meet with their manager 1-on-1 weekly during onboarding. A 30-day check-in with HR is mandatory.""",
        metadata={"source": "HR-005", "title": "Onboarding", "department": "hr"},
    ),
    Document(
        page_content="""Code of Conduct and Anti-Harassment. The company maintains a zero-tolerance policy for harassment, discrimination, or retaliation of any kind. Reports can be made to any manager, HR Business Partner, or anonymously through the EthicsPoint hotline at 1-800-555-0199. All reports are investigated within 5 business days. Retaliation against anyone making a good-faith report is itself a terminable offense. Annual anti-harassment training is mandatory for all employees and must be completed by November 30 each year.""",
        metadata={"source": "HR-006", "title": "Code of Conduct", "department": "hr"},
    ),
    Document(
        page_content="""Learning and Development Benefits. Each employee receives an annual Learning & Development budget of $2,000 for approved courses, conferences, certifications, and books. Requests must be pre-approved by the employee's manager. The company also offers tuition reimbursement of up to $5,250 per year for degree programs directly related to the employee's role. Employees must remain with the company for 12 months after course completion or repay a prorated amount. Conference attendance requires VP approval if travel is involved.""",
        metadata={"source": "HR-007", "title": "Learning and Development", "department": "hr"},
    ),
    Document(
        page_content="""Health and Wellness Benefits. The company offers three medical plan tiers: Basic PPO, Premium PPO, and HDHP with HSA. Employee premiums are 20%, 15%, and 10% of total cost respectively. Dental and vision are included at no cost to the employee. The wellness program provides a $50/month gym membership reimbursement, free access to the Headspace meditation app, and an annual $300 wellness stipend for fitness equipment, wearables, or mental health services. Employees can use wellness funds for therapy copays.""",
        metadata={"source": "HR-008", "title": "Health and Wellness", "department": "hr"},
    ),
    Document(
        page_content="""Offboarding and Departure Process. Resigning employees must give at least two weeks written notice through Workday. Managers notify HR and IT the same business day so access revocation can be scheduled. On the final day, employees return all company equipment tracked under the Asset Management Policy, including laptops, monitors, and security keys. Unreturned equipment over $500 in value is deducted from final pay where state law permits. Final pay, including accrued unused PTO up to the 5-day carryover cap, is issued within the next scheduled payroll run. Exit interviews are conducted by an HR Business Partner and are voluntary.""",
        metadata={"source": "HR-009", "title": "Offboarding", "department": "hr"},
    ),
    Document(
        page_content="""Internal Transfers and Promotions. Employees are eligible to apply for internal roles after 12 months in their current position. Transfers require approval from both the current and receiving manager, plus VP approval when the move crosses departments. Promotion decisions are made during the annual December review cycle and take effect the following January. Employees on an active Performance Improvement Plan are not eligible for transfer or promotion until the plan closes successfully. Compensation changes tied to promotion follow the Compensation Bands Policy and require Finance validation of budget availability.""",
        metadata={"source": "HR-010", "title": "Internal Transfers", "department": "hr"},
    ),
    Document(
        page_content="""Compensation Bands and Merit Increases. Every role maps to a compensation band with a defined minimum, midpoint, and maximum. Merit increase budgets are set during the annual budget cycle each October and are typically 3% of departmental payroll. Increases above 10% of current salary require VP approval; increases that would place an employee above their band maximum require CFO approval. Off-cycle adjustments are permitted only for promotions, market corrections, or retention cases documented in Workday. Managers cannot discuss individual band placements outside the review process.""",
        metadata={"source": "HR-011", "title": "Compensation Bands", "department": "hr"},
    ),
    Document(
        page_content="""Recruiting and Hiring Approvals. Every new position requires an approved headcount requisition in Workday before recruiting begins. Requisitions are approved by the hiring manager, the department VP, and Finance, which confirms the role is within the approved annual budget. Backfills for departed employees follow the same process but are expedited when the role was budgeted for the current fiscal year. Offers above the compensation band midpoint require VP approval; offers above the band maximum require CFO approval. All candidates complete a structured interview loop with at least four interviewers.""",
        metadata={"source": "HR-012", "title": "Recruiting and Hiring", "department": "hr"},
    ),
    Document(
        page_content="""Contingent Workers and Contractors. Contractors, temporary staff, and agency workers are engaged through the Procurement Policy and require a purchase order in Coupa before work begins. Engagements over $25,000 require VP approval; over $100,000 require CFO approval. Contractors do not receive employee benefits, PTO accrual, or Learning & Development budget. Any contractor engagement exceeding 12 months must be reviewed by HR and Legal for worker-classification risk. Contractor system access is provisioned with a fixed expiry date matching the contract end date.""",
        metadata={"source": "HR-013", "title": "Contingent Workers", "department": "hr"},
    ),
    Document(
        page_content="""Sick Leave and Short-Term Disability. Employees accrue 10 days of paid sick leave per year, separate from PTO. Sick leave does not carry over between calendar years. Absences longer than 3 consecutive days require a healthcare provider note submitted confidentially to HR. Short-term disability begins on day 8 of a qualifying absence and pays 60% of base salary for up to 12 weeks. Long-term disability begins after 12 weeks at 50% of base salary. Disability claims are administered by an external carrier; HR coordinates the intake but does not adjudicate claims.""",
        metadata={"source": "HR-014", "title": "Sick Leave and Disability", "department": "hr"},
    ),
    Document(
        page_content="""Bereavement and Emergency Leave. Employees receive 5 days of paid bereavement leave for an immediate family member (spouse, partner, child, parent, sibling) and 3 days for extended family. Additional unpaid leave may be approved by the employee's manager. Emergency leave of up to 3 paid days is available for natural disasters, home displacement, or urgent dependent care. Requests are logged in Workday but do not require advance notice. Travel costs related to bereavement are not reimbursable under the Corporate Travel Policy unless the company required the travel.""",
        metadata={"source": "HR-015", "title": "Bereavement Leave", "department": "hr"},
    ),
    Document(
        page_content="""Jury Duty and Civic Leave. The company provides paid leave for jury duty for up to 10 business days per calendar year. Employees forward the court summons to HR within 5 business days of receipt. Any jury stipend received from the court is retained by the employee and is not offset against pay. Voting leave of up to 2 hours is available where local law requires it. Military reserve duty follows applicable law and may extend beyond the 10-day civic leave cap with HR and VP approval.""",
        metadata={"source": "HR-016", "title": "Jury Duty and Civic Leave", "department": "hr"},
    ),
    Document(
        page_content="""Sabbatical Program. Employees with 5 years of continuous service are eligible for a 4-week paid sabbatical, and 6 weeks at 10 years. Sabbaticals require VP approval and must be scheduled at least 90 days in advance so coverage can be arranged. Sabbatical time does not replace PTO accrual, which continues during the sabbatical. Employees must return for at least 6 months following a sabbatical or repay a prorated portion, mirroring the clawback structure in the Learning and Development Policy. Sabbaticals cannot be split into increments shorter than two consecutive weeks.""",
        metadata={"source": "HR-017", "title": "Sabbatical Program", "department": "hr"},
    ),
    Document(
        page_content="""Retirement and 401(k) Benefits. The company matches 100% of employee 401(k) contributions up to 4% of base salary, with immediate vesting. Contributions are administered through the payroll provider and adjustments take effect the following pay period. Employees may change contribution rates at any time in Workday. The annual IRS contribution limit applies and is monitored by Payroll to prevent excess deferrals. Highly compensated employees may be subject to additional contribution limits following annual non-discrimination testing performed by Finance each January.""",
        metadata={"source": "HR-018", "title": "Retirement Benefits", "department": "hr"},
    ),
    Document(
        page_content="""Employee Assistance Program (EAP). The EAP provides 8 free confidential counseling sessions per issue per year for employees and household members. Services include mental health support, legal consultation, financial coaching, and dependent care referrals. EAP use is confidential and is never reported to managers or HR. The EAP is separate from the $300 annual wellness stipend and from medical plan therapy coverage, and employees may use all three. Crisis support is available 24/7 through the same hotline vendor that operates the EthicsPoint reporting line.""",
        metadata={"source": "HR-019", "title": "Employee Assistance Program", "department": "hr"},
    ),
    Document(
        page_content="""Employee Referral Program. Employees receive a $3,000 referral bonus for a successful hire into an engineering or sales role and $1,500 for all other roles. The referral must be logged in Workday before the candidate applies. Bonuses are paid after the referred employee completes 90 days of employment and are processed through payroll, not through Expensify. Hiring managers, recruiters, and anyone in the candidate's approval chain are ineligible for referral bonuses on that requisition. Referral bonuses are charged to the hiring department's budget.""",
        metadata={"source": "HR-020", "title": "Referral Program", "department": "hr"},
    ),
    Document(
        page_content="""Workplace Accommodations. Employees may request reasonable accommodations for disability, religious observance, or pregnancy through HR. Requests are handled confidentially and HR responds within 10 business days. Accommodation costs under $1,000 are approved by HR directly; above that threshold, department VP approval is required and the cost is charged to the department budget. Ergonomic equipment purchased as an accommodation is tracked under the Asset Management Policy but is exempt from the standard home office stipend limits in the Remote Work Policy.""",
        metadata={"source": "HR-021", "title": "Workplace Accommodations", "department": "hr"},
    ),
    Document(
        page_content="""Employee Data Privacy. Employee personal data is collected only for legitimate employment purposes and is stored in Workday with role-based access controls. Managers can view their direct reports' performance and compensation data but not medical, disability, or EAP records. Employees may request a copy of their personnel file once per calendar year. Data retention follows the Records Retention Policy: personnel records are kept for 7 years after termination. Any suspected breach of employee data must be reported immediately under the Incident Response Policy.""",
        metadata={"source": "HR-022", "title": "Employee Data Privacy", "department": "hr"},
    ),
    Document(
        page_content="""Conflicts of Interest. Employees must disclose any outside employment, board membership, or financial interest in a competitor, customer, or vendor. Disclosures are made annually and within 30 days of any new conflict arising. Employees may not participate in a vendor selection or purchase approval where they have a personal financial interest, which extends the segregation-of-duties requirements in the Procurement Policy. Gifts from vendors valued over $100 must be declined or disclosed. Violations are handled under the Code of Conduct.""",
        metadata={"source": "HR-023", "title": "Conflicts of Interest", "department": "hr"},
    ),
    Document(
        page_content="""Business Ethics and Anti-Bribery. The company prohibits offering or accepting anything of value to improperly influence a business decision, including facilitation payments. Anti-bribery training is mandatory annually for employees in sales, procurement, and finance roles and must be completed by November 30 alongside anti-harassment training. Any payment to a government official, regardless of amount, requires prior Legal approval. Suspected violations are reported through the EthicsPoint hotline and investigated within 5 business days under the Code of Conduct process.""",
        metadata={"source": "HR-024", "title": "Anti-Bribery", "department": "hr"},
    ),
    Document(
        page_content="""Workplace Safety and Incident Reporting. All workplace injuries must be reported to a manager and HR within 24 hours, regardless of severity. HR files required regulatory reports and coordinates workers' compensation claims. Remote employees are covered for injuries occurring in their designated home workspace during core working hours as defined in the Remote Work Policy. Annual safety training is required for employees working in labs or facilities. Serious incidents trigger an investigation led by HR and Facilities within 5 business days.""",
        metadata={"source": "HR-025", "title": "Workplace Safety", "department": "hr"},
    ),
    Document(
        page_content="""Travel Safety and Duty of Care. Employees traveling on company business are covered by the corporate travel insurance policy, which is arranged by Finance and applies automatically to trips booked through Navan. Travel to countries under an elevated advisory requires VP approval and a Legal risk review before booking. Employees must register international itineraries with HR so the company can reach them in an emergency. Medical evacuation coverage is included. Personal travel appended to a business trip is not covered and is not reimbursable under the Corporate Travel Policy.""",
        metadata={"source": "HR-026", "title": "Travel Safety", "department": "hr"},
    ),
    Document(
        page_content="""Relocation Assistance. New hires relocating more than 50 miles may receive relocation assistance of up to $10,000 for individual contributors and $20,000 for director-level and above. Relocation packages require VP approval and are funded from the hiring department's budget. Covered costs include movers, temporary housing up to 60 days, and one house-hunting trip booked through Navan under the Corporate Travel Policy. Employees who leave voluntarily within 12 months repay a prorated amount, consistent with the tuition reimbursement clawback in the Learning and Development Policy.""",
        metadata={"source": "HR-027", "title": "Relocation Assistance", "department": "hr"},
    ),
    Document(
        page_content="""Recognition and Spot Bonuses. Managers may award spot bonuses of up to $500 without additional approval; awards between $500 and $2,500 require VP approval, and anything above $2,500 requires CFO approval. Spot bonuses are processed through payroll and charged to the awarding manager's departmental budget. Non-cash recognition awards, such as gift cards, are taxable and must also be routed through payroll rather than reimbursed through Expensify. Annual recognition award budgets are set during the October budget cycle.""",
        metadata={"source": "HR-028", "title": "Recognition and Spot Bonuses", "department": "hr"},
    ),
    Document(
        page_content="""Volunteer Time Off. Employees receive 2 paid days per calendar year to volunteer with a registered nonprofit. Volunteer days are logged in Workday and do not carry over. The company matches employee charitable donations up to $1,000 per employee per year; matching requests are submitted to Finance with a donation receipt and processed through payroll. Volunteer time with organizations that create a conflict of interest, as defined in the Conflicts of Interest Policy, requires prior HR approval. Team volunteer events require manager approval and are charged to the department budget.""",
        metadata={"source": "HR-029", "title": "Volunteer Time Off", "department": "hr"},
    ),
    Document(
        page_content="""Grievance and Dispute Resolution. Employees who believe a policy has been applied unfairly may file a grievance with their HR Business Partner. HR acknowledges the grievance within 3 business days and completes a review within 15 business days. Employees may escalate to the department VP if unsatisfied with the outcome. Grievances alleging harassment, discrimination, or retaliation are redirected immediately into the Code of Conduct investigation process and are not handled through the standard grievance track. Retaliation for filing a grievance is a terminable offense.""",
        metadata={"source": "HR-030", "title": "Grievance Process", "department": "hr"},
    ),
    Document(
        page_content="""Attendance and Core Hours. Employees are expected to be available during core working hours of 10 AM to 3 PM in their assigned time zone, whether working onsite, hybrid, or fully remote. Schedule changes lasting more than two weeks require manager approval and an update in Workday. Repeated unexplained absence is addressed through the Performance Review process and may lead to a Performance Improvement Plan. Time zone changes lasting more than 30 days are treated as a relocation and require VP approval plus a payroll review for tax implications.""",
        metadata={"source": "HR-031", "title": "Attendance and Core Hours", "department": "hr"},
    ),
    Document(
        page_content="""Manager Responsibilities for Budget and Approvals. Managers are accountable for their team's spending against the departmental budget set in the October cycle. Managers approve expense reports in Expensify within 5 business days of submission, approve PTO requests in Workday, and confirm headcount requisitions before they route to Finance. Managers may not approve their own expenses, their manager's expenses, or any transaction where they have a conflict of interest. Approval authority cannot be delegated below the manager level without VP approval.""",
        metadata={"source": "HR-032", "title": "Manager Approval Duties", "department": "hr"},
    ),
    Document(
        page_content="""Training Compliance and Deadlines. Three trainings are mandatory annually and all share a November 30 completion deadline: anti-harassment for all employees, anti-bribery for sales, procurement, and finance roles, and security awareness for anyone with production system access. Completion is tracked in Workday and reported to the Audit Committee as part of SOX compliance evidence. Employees who miss the deadline lose access to non-essential systems until completion. Managers with team completion below 90% are flagged in their own performance review.""",
        metadata={"source": "HR-033", "title": "Training Compliance", "department": "hr"},
    ),
    Document(
        page_content="""Intellectual Property and Invention Assignment. Work product created within the scope of employment belongs to the company. Employees disclose inventions through the Legal team's invention disclosure process. Prior inventions must be listed at hire in the employment agreement to be excluded. Open-source contributions related to the employee's work require manager and Legal approval before publication. Employees may retain ownership of personal projects developed on personal time and equipment that do not relate to company business, subject to the Conflicts of Interest Policy.""",
        metadata={"source": "HR-034", "title": "Intellectual Property", "department": "hr"},
    ),
    Document(
        page_content="""Confidentiality and Non-Disclosure. All employees sign a confidentiality agreement at hire covering customer data, financial results, product roadmaps, and compensation structures. Unreleased financial results are material non-public information and are additionally governed by the Insider Trading Policy. Confidentiality obligations survive termination indefinitely. Sharing confidential information with contractors requires an executed NDA obtained through Legal before disclosure. Suspected disclosure incidents must be reported within 24 hours under the Incident Response Policy.""",
        metadata={"source": "HR-035", "title": "Confidentiality", "department": "hr"},
    ),
    Document(
        page_content="""Performance Improvement Plans. A Performance Improvement Plan runs 90 days and is required for employees rated 1 or 2 in the annual review, or at any point when a manager documents sustained underperformance. The plan specifies measurable objectives, weekly check-ins, and a defined outcome date. HR reviews every plan before it is issued. Employees on an active plan are ineligible for internal transfer, promotion, spot bonuses, or sabbatical. Successful completion closes the plan; failure results in separation handled through the Offboarding process.""",
        metadata={"source": "HR-036", "title": "Performance Improvement Plans", "department": "hr"},
    ),
    Document(
        page_content="""Employment Verification and References. Employment verification requests are handled exclusively by HR and confirm only title, dates of employment, and location. Salary information is released only with written employee consent or where legally required. Managers must not provide personal references on company letterhead or through company email. Requests from law enforcement or in response to a subpoena are routed to Legal before any response. Verification requests are completed within 5 business days.""",
        metadata={"source": "HR-037", "title": "Employment Verification", "department": "hr"},
    ),
    Document(
        page_content="""Company Equipment for Employees. Standard issue for every employee is a laptop, monitor, keyboard, mouse, and headset, provisioned by IT during the 30-day onboarding program. Equipment remains company property and is tracked under the Asset Management Policy. Requests for non-standard equipment above $1,000 require manager and VP approval and are purchased through Coupa rather than reimbursed through Expensify. Damaged equipment is replaced at no cost to the employee unless damage resulted from negligence. All equipment is returned during Offboarding.""",
        metadata={"source": "HR-038", "title": "Employee Equipment", "department": "hr"},
    ),
]

FIN_DOCS: list[Document] = [
    Document(
        page_content="""Expense Reimbursement Policy. Employees can be reimbursed for pre-approved business expenses. All expenses must be submitted via Expensify within 30 days of being incurred. Receipts are required for any single expense over $25. Meals during business travel are reimbursable up to $75 per day for domestic travel and $125 per day for international travel. Alcohol is not reimbursable except at client entertainment events with manager approval. Personal expenses incorrectly charged to company cards must be repaid within 14 days.""",
        metadata={"source": "FIN-001", "title": "Expense Reimbursement", "department": "finance"},
    ),
    Document(
        page_content="""Corporate Travel Policy. Business travel requires manager approval in advance. Flights: economy class for flights under 6 hours; premium economy allowed for flights over 6 hours; business class requires VP approval. Hotels must not exceed $250/night in standard US cities, $350/night in NYC/SF/LA, $400/night internationally. Ground transportation: Uber, Lyft, and rental cars are reimbursable. Rental car class is limited to mid-size. Personal vehicle mileage is reimbursed at the current IRS rate of $0.67 per mile.""",
        metadata={"source": "FIN-002", "title": "Travel Policy", "department": "finance"},
    ),
    Document(
        page_content="""Budget Planning and Approval. The annual budget cycle begins October 1st. Department heads submit proposed budgets by October 31st. Finance reviews and consolidates by November 30th. Final budget approval by the CFO and CEO occurs in the December board meeting. Quarterly budget reviews occur in the first week of each new quarter. Any spending exceeding 110% of an approved line item requires CFO approval. Capital expenditures over $50,000 require board approval regardless of budget status.""",
        metadata={"source": "FIN-003", "title": "Budget Planning", "department": "finance"},
    ),
    Document(
        page_content="""Accounts Payable (AP) Process. Vendor invoices should be submitted to ap@company.com. Standard payment terms are Net 30 from invoice date. Early payment discounts of 2/10 Net 30 are negotiated with strategic vendors. All invoices over $10,000 require dual approval: the budget owner and a Finance manager. Invoices over $100,000 additionally require CFO approval. Vendors must be set up in Coupa with a completed W-9 (domestic) or W-8 (international) before their first invoice can be processed.""",
        metadata={"source": "FIN-004", "title": "Accounts Payable", "department": "finance"},
    ),
    Document(
        page_content="""Revenue Recognition Policy. The company follows ASC 606 for revenue recognition. Subscription revenue is recognized ratably over the contract term. Professional services revenue is recognized as services are delivered based on a percentage-of-completion method. One-time setup fees are deferred and recognized over the expected customer life of 3 years. Any contract modifications require review by the Revenue Operations team. Channel partner commissions are recognized at the point of contract signing, not collection.""",
        metadata={"source": "FIN-005", "title": "Revenue Recognition", "department": "finance"},
    ),
    Document(
        page_content="""Corporate Card Program. Employees in roles requiring frequent business expenses are eligible for a corporate American Express card. Card applications are approved by the employee's manager and Finance. Monthly reconciliation in Expensify is required by the 5th of the following month. Cards not reconciled within 60 days are automatically suspended. Cash advances are not permitted on corporate cards. Lost or stolen cards must be reported immediately to Amex (1-800-528-4800) and the corporate card administrator.""",
        metadata={"source": "FIN-006", "title": "Corporate Card", "department": "finance"},
    ),
    Document(
        page_content="""Financial Reporting and Close. The monthly close process runs from the 1st through the 5th business day of the following month. The Finance team publishes the preliminary P&L by day 6 and the board-ready financial package by day 10. Quarterly financial statements are prepared according to GAAP and reviewed by external auditors (KPMG). The fiscal year ends December 31. Annual audit typically completes in late February with the 10-K filed in March. SOX compliance controls apply to all financial processes.""",
        metadata={"source": "FIN-007", "title": "Financial Reporting", "department": "finance"},
    ),
    Document(
        page_content="""Procurement and Purchase Orders. All purchases over $5,000 require a formal purchase order (PO) created in Coupa before the order is placed. Vendors must be in good standing and properly set up. For purchases between $5,000 and $25,000, the budget owner and direct manager approve. Between $25,000 and $100,000, VP approval is required. Above $100,000, CFO approval is required. Sole-source justifications must accompany any PO that did not go through competitive bidding when value exceeds $50,000.""",
        metadata={"source": "FIN-008", "title": "Procurement", "department": "finance"},
    ),
    Document(
        page_content="""Travel Booking and the Navan Platform. All business travel must be booked through Navan, the corporate travel platform, so that duty-of-care tracking and negotiated rates apply. Bookings made outside Navan require a written exception from Finance and may be reimbursed only at the equivalent Navan fare. Trips must be booked at least 14 days in advance where practical; bookings inside 7 days require manager approval because of fare premiums. Cancellations must be processed in Navan to recover refundable value. Unused tickets are tracked centrally and applied to future trips.""",
        metadata={"source": "FIN-009", "title": "Travel Booking", "department": "finance"},
    ),
    Document(
        page_content="""Client Entertainment and Meals. Client entertainment requires manager approval in advance and must have a documented business purpose with the names and companies of all attendees. Entertainment is capped at $150 per attendee. Alcohol is reimbursable only at client entertainment events, unlike ordinary business travel meals which exclude it under the Expense Reimbursement Policy. Entertainment of government officials is prohibited without prior Legal approval under the Anti-Bribery Policy. Receipts are required regardless of amount for all entertainment expenses.""",
        metadata={"source": "FIN-010", "title": "Client Entertainment", "department": "finance"},
    ),
    Document(
        page_content="""Payroll Processing. Employees are paid semi-monthly on the 15th and the last business day of each month. Payroll changes, including new hires, terminations, and compensation adjustments, must be entered in Workday by the 5th and the 20th respectively to make the corresponding run. Off-cycle payments are issued only for missed wages or legally required final pay. Spot bonuses, referral bonuses, and taxable gift cards are processed through payroll rather than Expensify. Payroll registers are reviewed and signed off by a Finance manager as a SOX control.""",
        metadata={"source": "FIN-011", "title": "Payroll Processing", "department": "finance"},
    ),
    Document(
        page_content="""Asset Management and Depreciation. Any single item purchased for $2,500 or more is capitalized and depreciated over its useful life: 3 years for laptops and IT equipment, 5 years for furniture, and the lease term for leasehold improvements. Items below the threshold are expensed in the period incurred. All capitalized assets are tagged and tracked in the fixed asset register maintained by Finance. Employee equipment issued under the HR equipment policy is recorded here and reconciled during Offboarding. Asset disposals require Finance approval and are recorded with a gain or loss.""",
        metadata={"source": "FIN-012", "title": "Asset Management", "department": "finance"},
    ),
    Document(
        page_content="""Capital Expenditure Requests. Capital expenditure requests are submitted during the October budget cycle with a business case including expected payback period. Unbudgeted capital requests require CFO approval, and any capital expenditure over $50,000 requires board approval regardless of whether it was budgeted. Capital projects over $250,000 require a post-implementation review by Finance within 12 months of completion. Capital and operating expenses may not be reclassified after the period closes without Controller approval and an audit trail.""",
        metadata={"source": "FIN-013", "title": "Capital Expenditures", "department": "finance"},
    ),
    Document(
        page_content="""Vendor Management and Due Diligence. New vendors are onboarded in Coupa with a completed W-9 or W-8, banking details verified by a callback to a known contact, and a sanctions screening performed by Finance. Vendors handling company or customer data additionally require a security review and an executed data processing agreement through Legal. Vendor contracts over $100,000 require CFO approval and Legal review. Vendor performance is reviewed annually for strategic vendors. Changes to vendor banking details always require a second-person verification as an anti-fraud control.""",
        metadata={"source": "FIN-014", "title": "Vendor Management", "department": "finance"},
    ),
    Document(
        page_content="""Contract Signature Authority. Contracts up to $25,000 may be signed by a department VP. Contracts between $25,000 and $100,000 require CFO signature. Contracts above $100,000 require both CFO and CEO signature. Any contract with a term longer than 36 months, an auto-renewal clause, or a non-standard indemnity requires Legal review regardless of value. No employee may sign a contract that commits the company to spending outside their approved budget. Signature authority cannot be delegated without written CFO approval.""",
        metadata={"source": "FIN-015", "title": "Signature Authority", "department": "finance"},
    ),
    Document(
        page_content="""SOX Compliance and Internal Controls. The company maintains documented internal controls over financial reporting under Sarbanes-Oxley. Key controls include segregation of duties between purchase approval and payment execution, dual approval for invoices over $10,000, monthly account reconciliations, and quarterly access reviews for financial systems. Control owners certify their controls quarterly. Deficiencies are logged and remediated within the quarter. External auditors test key controls annually as part of the audit that completes in late February.""",
        metadata={"source": "FIN-016", "title": "SOX Compliance", "department": "finance"},
    ),
    Document(
        page_content="""Internal Audit Function. Internal Audit reports to the Audit Committee of the board and operates independently of Finance management. The annual audit plan is risk-based and approved by the Audit Committee each January. Audits cover expense compliance, procurement, revenue recognition, and access controls. Findings are rated critical, high, medium, or low; critical and high findings require a remediation plan within 30 days. Management responses are tracked to closure. Internal Audit also reviews the training completion evidence collected in Workday.""",
        metadata={"source": "FIN-017", "title": "Internal Audit", "department": "finance"},
    ),
    Document(
        page_content="""Treasury and Banking. The company maintains operating accounts with two banking partners to limit counterparty risk. Only the CFO and Treasurer are authorized signers on corporate accounts. Wire transfers over $50,000 require dual authorization, and any wire to a new beneficiary requires a verification callback consistent with the Vendor Management anti-fraud control. Excess cash is invested only in instruments approved by the board investment policy. Foreign currency exposure over $1,000,000 is reviewed quarterly for hedging.""",
        metadata={"source": "FIN-018", "title": "Treasury and Banking", "department": "finance"},
    ),
    Document(
        page_content="""Tax Compliance. The company files federal, state, and local returns with support from an external tax advisor. Sales tax nexus is evaluated quarterly as the company adds employees in new states, which makes employee relocations and long-term remote work arrangements a tax matter as well as an HR one. Contractor payments over $600 annually generate a 1099. International contractor payments require a W-8 on file before payment. R&D tax credit documentation is collected annually from Engineering. Tax provision entries are reviewed by the Controller during quarterly close.""",
        metadata={"source": "FIN-019", "title": "Tax Compliance", "department": "finance"},
    ),
    Document(
        page_content="""Insurance Coverage. The company maintains general liability, directors and officers, cyber liability, errors and omissions, workers' compensation, and corporate travel insurance. Travel insurance applies automatically to trips booked through Navan and includes medical evacuation. Cyber liability coverage requires that the company maintain the security controls described in the Information Security Policy; lapses can void coverage. Certificates of insurance are issued by Finance on request, typically within 5 business days. Policies renew annually on July 1.""",
        metadata={"source": "FIN-020", "title": "Insurance Coverage", "department": "finance"},
    ),
    Document(
        page_content="""Software and SaaS Purchasing. All software purchases, regardless of value, require review by IT and Security before a purchase order is issued in Coupa, because SaaS tools frequently process company or customer data. Purchases over $5,000 follow the standard Procurement thresholds. Annual subscriptions over $25,000 require VP approval and a documented renewal date so Finance can plan the renewal. Employees may not expense software through Expensify to bypass procurement review. Shadow IT discovered during access reviews is either onboarded properly or terminated.""",
        metadata={"source": "FIN-021", "title": "Software Purchasing", "department": "finance"},
    ),
    Document(
        page_content="""Information Security Controls. Access to production and financial systems is granted on a least-privilege basis and reviewed quarterly as a SOX control. Multi-factor authentication is mandatory for all systems handling financial or customer data. Security awareness training is required annually for anyone with production access and shares the November 30 deadline with other mandatory training. Contractor access is provisioned with an expiry date matching the contract term. Failure to maintain these controls can void the cyber liability insurance coverage.""",
        metadata={"source": "FIN-022", "title": "Information Security", "department": "finance"},
    ),
    Document(
        page_content="""Incident Response. Suspected security incidents, data breaches, or losses of company data must be reported within 24 hours to the security team, matching the reporting window for confidentiality incidents in the HR policy. The incident commander assesses severity and convenes Legal, Finance, and Communications for high-severity events. Incidents involving employee personal data trigger notification obligations coordinated with HR under the Employee Data Privacy Policy. Post-incident reviews are completed within 15 business days and findings are tracked by Internal Audit.""",
        metadata={"source": "FIN-023", "title": "Incident Response", "department": "finance"},
    ),
    Document(
        page_content="""Records Retention. Financial records are retained for 7 years, matching the retention period for personnel records under the Employee Data Privacy Policy. Contracts are retained for 7 years after expiration. Tax records are retained for 7 years after filing. Email is retained for 3 years unless subject to a legal hold. Legal holds are issued by the Legal team and suspend all deletion for the affected records until released. Destruction of records outside the retention schedule requires Controller approval and is documented.""",
        metadata={"source": "FIN-024", "title": "Records Retention", "department": "finance"},
    ),
    Document(
        page_content="""Insider Trading Policy. Employees may not trade company securities while in possession of material non-public information, including unreleased financial results governed by the Confidentiality Policy. Trading windows open two full business days after quarterly earnings are released and close two weeks before quarter end. Designated insiders, including all VPs and above and the Finance team, must pre-clear trades with Legal. The policy applies to family members and household accounts. Violations may result in termination and regulatory referral.""",
        metadata={"source": "FIN-025", "title": "Insider Trading", "department": "finance"},
    ),
    Document(
        page_content="""Intercompany and Transfer Pricing. Transactions between company entities are conducted at arm's length and documented under the transfer pricing policy prepared annually with the external tax advisor. Intercompany balances are reconciled monthly during close and must net to zero at consolidation. Cross-border service charges use a cost-plus methodology. Any new intercompany arrangement requires Controller and tax advisor review before the first transaction. Intercompany agreements are executed under the Signature Authority Policy.""",
        metadata={"source": "FIN-026", "title": "Transfer Pricing", "department": "finance"},
    ),
    Document(
        page_content="""Purchasing Cards and Petty Cash. Purchasing cards are issued to Facilities and Office Management for recurring low-value purchases under $500. Petty cash is limited to $200 per location and is reconciled monthly. Purchasing card transactions are reconciled in Expensify by the 5th of the following month, matching the corporate card deadline. Purchasing cards may not be used to circumvent the $5,000 purchase order threshold by splitting a larger purchase into smaller transactions, which is an audit finding whenever detected.""",
        metadata={"source": "FIN-027", "title": "Purchasing Cards", "department": "finance"},
    ),
    Document(
        page_content="""Chargebacks and Departmental Cost Allocation. Shared costs such as software licenses, facilities, and IT support are allocated to departments monthly based on headcount. Departments see allocations in their monthly budget-to-actual report published by day 10 of close. Disputes about an allocation must be raised with Finance within 30 days of the report. Referral bonuses, spot bonuses, relocation, and accommodation costs above $1,000 are charged directly to the originating department rather than allocated. Allocation methodology changes require CFO approval.""",
        metadata={"source": "FIN-028", "title": "Cost Allocation", "department": "finance"},
    ),
    Document(
        page_content="""Fraud Prevention and Whistleblower Protection. Suspected financial fraud is reported through the same EthicsPoint hotline used for Code of Conduct reports and is investigated jointly by Internal Audit and Legal. Common risk patterns monitored include duplicate invoices, altered vendor banking details, split purchases below approval thresholds, and expense reports submitted just under the $25 receipt requirement. Whistleblowers are protected from retaliation, which is a terminable offense. The Audit Committee receives a summary of all fraud reports quarterly.""",
        metadata={"source": "FIN-029", "title": "Fraud Prevention", "department": "finance"},
    ),
    Document(
        page_content="""Purchase Requisition to Payment Cycle. The end-to-end cycle runs requisition, approval, purchase order, goods receipt, invoice, three-way match, and payment. The three-way match compares the purchase order, the receipt, and the invoice before payment is released; mismatches over 5% are held for review. Segregation of duties requires that the person approving the purchase is never the person releasing the payment. Payments run twice weekly on Tuesdays and Thursdays under Net 30 terms. Emergency payments require CFO approval.""",
        metadata={"source": "FIN-030", "title": "Requisition to Payment", "department": "finance"},
    ),
    Document(
        page_content="""Foreign Currency and International Expenses. Expenses incurred in foreign currency are reimbursed at the exchange rate on the transaction date as recorded by the corporate card provider, or at the published month-end rate for cash expenses. Employees should use the corporate card abroad to avoid conversion disputes. Foreign transaction fees are reimbursable. International travel meal per diems are $125 per day, higher than the $75 domestic rate, and international hotel caps are $400 per night under the Corporate Travel Policy.""",
        metadata={"source": "FIN-031", "title": "Foreign Currency Expenses", "department": "finance"},
    ),
    Document(
        page_content="""Grants, Rebates, and Incentives. Government grants, utility rebates, and vendor incentives are recorded as a reduction of the related expense rather than as revenue. Any grant with performance conditions is deferred until the conditions are met, consistent with ASC 606 principles applied in the Revenue Recognition Policy. Applications for grants over $100,000 require CFO approval before submission because of the compliance obligations they create. Grant compliance reporting is tracked by Finance and reviewed by Internal Audit.""",
        metadata={"source": "FIN-032", "title": "Grants and Rebates", "department": "finance"},
    ),
    Document(
        page_content="""Facilities and Office Spend. Office leases require CFO and board approval given their capital commitment and multi-year term. Office supplies and pantry purchases are made on purchasing cards under the $500 limit. Office improvements above $2,500 are capitalized as leasehold improvements and depreciated over the lease term under the Asset Management Policy. Facilities coordinates with HR on workplace safety inspections and with IT on equipment staging for the 30-day onboarding program. Desk moves and space planning are approved by Facilities.""",
        metadata={"source": "FIN-033", "title": "Facilities Spend", "department": "finance"},
    ),
    Document(
        page_content="""Financial Planning and Forecasting. Finance produces a rolling 12-month forecast updated monthly after close. Department heads submit headcount and spending assumptions by day 12 of each month. Variances greater than 10% against forecast require a written explanation from the department head. The forecast feeds the quarterly board package and informs whether headcount requisitions are approved. Scenario planning for revenue downside cases is refreshed quarterly and reviewed with the CEO and the board.""",
        metadata={"source": "FIN-034", "title": "Financial Planning", "department": "finance"},
    ),
    Document(
        page_content="""Customer Billing and Collections. Invoices are issued on contract signature for annual prepay customers and monthly in arrears for usage-based contracts. Standard customer payment terms are Net 30, matching vendor terms under the Accounts Payable Policy. Accounts more than 60 days past due are escalated to the account owner and Finance jointly; accounts over 90 days past due are suspended pending payment. Bad debt write-offs over $25,000 require CFO approval. Disputed invoices are documented and tracked until resolved by Revenue Operations.""",
        metadata={"source": "FIN-035", "title": "Billing and Collections", "department": "finance"},
    ),
    Document(
        page_content="""Sales Commissions. Commissions are calculated monthly based on booked and signed contracts and are paid through payroll in the month following booking. Channel partner commissions are recognized at contract signing rather than collection under the Revenue Recognition Policy. Commission plans are approved annually by the CFO and the VP of Sales during the October budget cycle. Clawbacks apply when a customer cancels within 90 days of signing or fails to pay within 120 days. Commission disputes are raised with Finance within 30 days of the payment.""",
        metadata={"source": "FIN-036", "title": "Sales Commissions", "department": "finance"},
    ),
    Document(
        page_content="""Mergers, Acquisitions, and Investments. Any acquisition, minority investment, or divestiture requires board approval regardless of size. Finance leads financial due diligence with support from Legal on contracts and HR on employee census, benefits harmonization, and retention planning. Acquired entities are integrated into the monthly close within two quarters of closing. Purchase price allocation is prepared with the external auditors. Retention bonuses for acquired employees are approved by the CFO and processed through payroll rather than Expensify.""",
        metadata={"source": "FIN-037", "title": "Mergers and Acquisitions", "department": "finance"},
    ),
]

#: Keyed to match `INDEX_CONFIG` in `app.retrieval.rag`; declared locally so the
#: corpus module stays free of retrieval imports.
DOCUMENTS: dict[Literal["hr", "finance"], list[Document]] = {"hr": HR_DOCS, "finance": FIN_DOCS}
