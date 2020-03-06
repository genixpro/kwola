import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import Typography from '../../../components/uielements/typography/index.js';
import Icon from '../../../components/uielements/icon/index.js';
import ExpansionPanel, {
  ExpansionPanelDetails,
  ExpansionPanelSummary,
  ExpansionPanelActions,
} from '../../../components/uielements/expansionPanel';
import Chip from '../../../components/uielements/chips';
import Button from '../../../components/uielements/button';
import Divider from '../../../components/uielements/dividers';

function DetailedExpansionPanel(props) {
  const { classes } = props;
  return (
    <div className={classes.root}>
      <ExpansionPanel defaultExpanded>
        <ExpansionPanelSummary expandIcon={<Icon>expand_more</Icon>}>
          <div className={classes.column}>
            <Typography className={classes.heading}>Location</Typography>
          </div>
          <div className={classes.column}>
            <Typography className={classes.secondaryHeading}>
              Select trip destination
            </Typography>
          </div>
        </ExpansionPanelSummary>
        <ExpansionPanelDetails className={classes.details}>
          <div className={classes.column} />
          <div className={classes.column}>
            <Chip
              label="Barbados"
              className={classes.chip}
              onDelete={() => {}}
            />
          </div>
          <div className={classNames(classes.column, classes.helper)}>
            <Typography type="caption">
              Select your destination of choice
              <br />
              <a href="#sub-labels-and-columns" className={classes.link}>
                Learn more
              </a>
            </Typography>
          </div>
        </ExpansionPanelDetails>
        <Divider />
        <ExpansionPanelActions>
          <Button size="small">Cancel</Button>
          <Button size="small" color="primary">
            Save
          </Button>
        </ExpansionPanelActions>
      </ExpansionPanel>
    </div>
  );
}

DetailedExpansionPanel.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default DetailedExpansionPanel;
