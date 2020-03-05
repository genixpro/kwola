import React from 'react';
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import SingleMail from '../singleMail';
import MailAction from '../singleMailActions';
import HelperText from '../../utility/helper-text';
import { timeDifference } from '../../../helpers/utility';
import MailListWrapper, {
  MailListItem,
  Checkbox,
  CheckboxWrapper,
  MailListInfos,
  Avatar,
  Icon,
  ExpansionPanel,
} from './mailList.style';
import {
  ExpansionPanelSummary,
  ExpansionPanelDetails,
} from '../../uielements/expansionPanel';

const theme = createMuiTheme({
  overrides: {
    MuiExpansionPanelSummary: {
      expandIcon: {
        display: 'none',
      },
      content: {
        width: '100%',
        margin: '0',
      },
    },
    MuiExpansionPanelDetails: {
      root: {
        padding: '0',
      },
    },
  },
});

export default ({
  mails,
  selectMail,
  selectedMail,
  toggleListVisible,
  replyMail,
  changeReplyMail,
  changeComposeMail,
  bulkActions,
  checkedMails,
  updateCheckedMail,
  selectedBucket,
  hideSearchbar,
  ...props
}) => {
  const activeClass = !hideSearchbar ? 'cH3cKed' : '';
  const renderSingleMail = (mail, key) => {
    const isSelected = selectedMail === mail.id;
    const checked = checkedMails[mail.id] || false;
    const recpName = mail.name;
    const signature = {
      splitLet: recpName
        .match(/\b(\w)/g)
        .join('')
        .split('', 2),
    };
    // const activeClass = isSelected ? 'activeMail' : '';
    const unreadMail = !mail.read ? 'unread ' : '';

    const expandClass = isSelected ? 'eXpaNd3d ' : '';

    const checkedItem = checked ? 's3lecTed ' : '';

    const onMailClick = () => {
      if (isSelected) {
        selectMail();
      } else {
        selectMail(mail.id);
        if (toggleListVisible) {
          toggleListVisible();
        }
      }
    };

    return (
      <ThemeProvider theme={theme} key={key}>
        <ExpansionPanel
          expanded={isSelected}
          onChange={onMailClick}
          className="expansionPanel"
        >
          <ExpansionPanelSummary className="expansionPanelSummary">
            <MailListItem
              onClick={onMailClick}
              className={`${checkedItem}${expandClass}${unreadMail}`}
            >
              <CheckboxWrapper onClick={event => event.stopPropagation()}>
                <Checkbox
                  checked={checked}
                  color="primary"
                  onChange={() => {
                    checkedMails[mail.id] = !checked;
                    updateCheckedMail(checkedMails);
                  }}
                />

                <Avatar style={{ backgroundColor: selectedBucket }}>
                  {mail.img ? (
                    <img alt="#" src={mail.img} />
                  ) : (
                    <span className="sPl1t-l3t">{signature.splitLet}</span>
                  )}
                  <Icon>done</Icon>
                </Avatar>
              </CheckboxWrapper>

              <MailListInfos>
                <h3 className="name">{mail.name}</h3>
                <div className="subDesc">
                  <p className="subject">{mail.subject}</p>
                  <p className="description">{mail.body}</p>
                </div>
                <span className="mailDate">{timeDifference(mail.date)}</span>
              </MailListInfos>
              <MailAction
                mail={mail}
                filterMails={mails}
                selectMail={selectMail}
                toggleListVisible={toggleListVisible}
                bulkActions={bulkActions}
              />
            </MailListItem>
          </ExpansionPanelSummary>
          <ExpansionPanelDetails>
            {isSelected && (
              <SingleMail
                mails={mails}
                replyMail={replyMail}
                changeReplyMail={changeReplyMail}
                selectMail={selectMail}
                selectedMail={selectedMail}
                changeComposeMail={changeComposeMail}
                toggleListVisible={toggleListVisible}
                bulkActions={bulkActions}
                {...props}
              />
            )}
          </ExpansionPanelDetails>
        </ExpansionPanel>
      </ThemeProvider>
    );
  };

  return (
    <MailListWrapper className={`${activeClass}`}>
      {mails.length === 0 ? <HelperText text="No mail found" /> : ''}
      {mails.map((mail, index) => renderSingleMail(mail, index))}
    </MailListWrapper>
  );
};
