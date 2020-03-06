import React from 'react';
import PropTypes from 'prop-types';
import Lists, {
  ListItem,
  ListItemText,
  ListSubheader,
} from '../../../components/uielements/lists';

function PinnedSubheaderList(props) {
  const { classes } = props;

  return (
    <Lists className={classes.root}>
      {[0, 1, 2, 3, 4].map(sectionId => (
        <div key={`section-${sectionId}`} className={classes.listSection}>
          <ListSubheader>{`I'm sticky ${sectionId}`}</ListSubheader>
          {[0, 1, 2].map(item => (
            <ListItem button key={`item-${sectionId}-${item}`}>
              <ListItemText primary={`Item ${item}`} />
            </ListItem>
          ))}
        </div>
      ))}
    </Lists>
  );
}

PinnedSubheaderList.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default PinnedSubheaderList;
